import numpy as np
import torch
from torch import tensor, float32
from src.algorithms.Agent import Agent
from src.utils import Basis, Critic, Policy, utils
import torch.functional as F
import wandb

"""
NOTE
- Separate out the state-feature construction part
"""


class ActorCritic2Potential(Agent):
    def __init__(self, config):
        super(ActorCritic2Potential, self).__init__(config)

        # Get state features and instances for Actor, Value, Reward, and Gamma functions
        config.state_lr = config.actor_lr
        self.state_features_actor = Basis.get_Basis(config=config)
        # print(config)
        config.state_lr = config.actor_lr * config.critic_lr_ratio
        self.state_features_critic = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features_actor.feature_dim,
                                                                     config=config)


        self.critic = Critic.Critic(state_dim=self.state_features_critic.feature_dim, config=config)
        self.memory = utils.TrajectoryBuffer(buffer_size=1, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)
        
        self.modules = [('actor', self.actor), ('state_features_actor', self.state_features_actor), ('critic', self.critic), ('state_features_critic', self.state_features_critic)]
        # aux_weight only a parameter for Reinforce
        self.aux_weight = config.aux_weight
        self.running_return_outer = None
        self.running_return_aux = None

        self.counter = 0
        self.init()

    def reset(self):
        super(ActorCritic2Potential, self).reset()
        self.memory.next()
        self.counter += 1
        self.gamma_t = 1

    def get_action(self, state, explore=0):
        explore = 0  # Don't do eps-greedy with policy gradients
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features_actor.forward(state.view(1, -1))
        # prob is 1 in continus action case.
        action, prob, dist = self.actor.get_action(state, explore=explore)

        if self.config.debug:
            self.track_entropy(dist, action)

        return action, prob
    def get_aligned_reward_terms(self, states):
        return None, None
    
    def get_aligned_return_gamma(self):
        with torch.no_grad():
            s, a, beta, r, aux_r, mask = self.memory.sample_last(0)
            B, H, D = s.shape
            _, _, A = a.shape

            # create state features
            s_feature_actor = self.state_features_actor.forward(s.view(B * H, D))  # BxHxD -> (BxH)xd
            s_feature_critic = self.state_features_critic.forward(s.view(B * H, D))  # BxHxD -> (BxH)xd
            a = a.view(B * H, A)  # BxHxA -> (BxH)xA
            mask = mask.view(B * H, -1)
            r = r.view(B * H, -1)  # BxH -> (BxH)x1
            aux_r = aux_r.view(B * H, -1)
            # not using the aux reward
            # r = r + aux_r
            r_aux = self.aux_weight * aux_r 
            # Potential based reward shaping
            r[:-1] = r[:-1] + (self.config.gamma * r_aux[1:] - r_aux[:-1])
            # r = r + self.aux_weight * aux_r
            gamma = self.config.gamma
            gamma = 1.0 # use a # gamma of one for this case. 
            r = r * mask
            r = r.view(B, H)
            # mask = mask.view(B, H)
            returns = torch.zeros_like(r)
            returns[:, H - 1] = r[:, H - 1]
            gamma_weights = torch.ones((1, H))
            for i in range(H - 2, -1, -1):
                returns[:, i] = r[:, i] + gamma * returns[:, i + 1].clone()
            index_end = (torch.argmin(mask) - 1).item()
            return torch.mean(returns[0,: index_end+1]).item(), gamma

    def update(self, s1, a1, prob, r1, aux_r1, s2, done):
        # Batch episode history
        self.memory.add(s1, a1, prob, r1, aux_r1)
        # self.memory.add(s1, a1, prob, self.gamma_t * r1, aux_r1)
        # self.gamma_t *= self.config.gamma
        if done:
            self.optimize()

    def optimize(self):
        # print(self.memry)
        s, a, beta, r, aux_r, mask = self.memory.sample_last(0)  # BxHxD, BxHxA, BxH, BxH, BxH, BxH

        # Not using aux_r to optimize stuff
        # r = r + aux_r  # Combine both primary and the auxiliary rewards
        r_aux = self.aux_weight * aux_r
        # r = r + self.aux_weight * aux_r
        # print(r)
        B, H, D = s.shape
        _, _, A = a.shape
        r_aux = r_aux.view(B * H, -1)  # BxH -> (BxH)x1
        r = r.view(B * H, -1)  # BxH -> (BxH)x1
        r[:-1] = r[:-1] + (self.config.gamma * r_aux[1:] - r_aux[:-1])
        # create state features
        # s_feature = self.state_features.forward(s.view(B * H, D))  # BxHxD -> (BxH)xd
        s_feature_actor = self.state_features_actor.forward(s.view(B * H, D))  # BxHxD -> (BxH)xd
        s_feature_critic = self.state_features_critic.forward(s.view(B * H, D))  # BxHxD -> (BxH)xd
        # print(s_feature.sum())
        a = a.view(B * H, A)  # BxHxA -> (BxH)xA
        # r = r.view(B * H, -1)  # BxH -> (BxH)x1
        mask = mask.view(B * H, -1)  # BxH -> (BxH)x1

        # ------ optimize critic -------- 
        val_pred = self.critic.forward(s_feature_critic)                           # (BxH)xd -> (BxH)x1
        vals = val_pred.detach()    # Detach targets from grad computation.

        # For each epsiode the structure of s and mask are
        # s_feats = [s0, s1, ......s_\infty, s0, s1, ...]
        # mask    = [1, 1, ......., 0, 1, 1, ...] 
        val_exp = torch.zeros_like(vals)                                    # (BxH)x1
        val_exp[:-1] = r[:-1] +  self.config.gamma * vals[1:] * mask[1:] 

        # loss_critic = F.smooth_l1_loss(val_pred, val_exp)
        # loss_critic = F.mse_loss(val_pred, val_exp)
        loss_critic = torch.sum(mask * (val_exp - val_pred)**2)/torch.sum(mask)  # Don't do torch.mean(), it will scre up because of unused vals in between
        
        # Update the Critic
        # self.step(loss_critic)

        # ---------------------- optimize actor ----------------------
        # Get action probabilities and TD error
        log_pi, _ = self.actor.get_logprob_dist(s_feature_actor, a)                   # (BxH)xd, (BxH)xA -> (BxH)x1
        # calculate stuff again
        # val_pred = self.critic.forward(s_feature).detach()
        # vals_exp = torch.zeros_like(val_pred)
        # val_exp[:-1] = r[:-1] +  self.config.gamma * val_pred[1:] * mask[1:]
        td_error = (val_exp - val_pred).detach()                                # (BxH)x1 - (BxH)x1

        # Reshape
        td_error = td_error.view(B, H)
        log_pi = log_pi.view(B, H)                                                  # (BxH)x1 -> BxH
        mask = mask.view(B, H)

        # gamma_weights = torch.tensor([[self.config.gamma**i for i in range(H)]])      # gamma_weights to get exact policy grad. 

        # compute policy grad
        # mean is fine here (instead of sum) as trajectry length is fixed (artifically)
        # masking is needed here as td_error need not be zero for ghost states in the buffer
        log_pi_td = torch.mean(mask * log_pi * td_error, dim=-1, keepdim=True)   # mean(BxH * BxH) -> Bx1

        # Compute the final loss
        loss_actor = -1.0 * torch.mean(log_pi_td)                                      # mean(Bx1) -> 1
        # wandb.log({"loss_actor": loss_actor.item(), "loss_critic": loss_critic.item()})
        # --------------------------------------------------------
        # update the actor
        # self.step(loss_actor)
        loss = loss_critic + loss_actor   
        self.step(loss)
        
        with torch.no_grad():
            r = r.view(B, H)
            r = r * mask
            returns = torch.zeros_like(r)
            returns[:, H - 1] = r[:, H - 1]
            gamma_weights = torch.ones((1, H))
            for i in range(H - 2, -1, -1):
                returns[:, i] = r[:, i] + self.config.gamma * returns[:, i + 1].clone()

            if self.running_return_outer is None:
                self.running_return_outer = returns.mean().item()
            else:
                self.running_return_outer = self.running_return_outer * 0.99 + 0.01 * returns.mean().item()



        


