import numpy as np
import torch
from torch import tensor, float32
from src.algorithms.Agent import Agent
from src.utils import Basis, Critic, Policy, utils
import torch.nn.functional as F
import wandb

"""
TODO
- Disable actor L2 here
"""


class ActorCriticOnline(Agent):
    def __init__(self, config):
        super(ActorCriticOnline, self).__init__(config)

        # Get state features and instances for Actor, Value, Reward, and Gamma functions
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features.feature_dim,
                                                                     config=config)


        self.critic = Critic.Critic(state_dim=self.state_features.feature_dim, config=config)
        self.memory = utils.TrajectoryBuffer(buffer_size=1, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)
        
        self.modules = [('actor', self.actor), ('state_features', self.state_features), ('critic', self.critic)]
        # aux_weight only a parameter for Reinforce
        self.aux_weight = config.aux_weight
        self.running_return_outer = None
        self.running_return_aux = None

        self.counter = 0
        self.init()

    def reset(self):
        super(ActorCriticOnline, self).reset()
        self.memory.next()
        self.counter += 1
        self.gamma_t = 1

    def get_action(self, state, explore=0):
        explore = 0  # Don't do eps-greedy with policy gradients
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
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
            s_feature = self.state_features.forward(s.view(B * H, D))  # BxHxD -> (BxH)xd
            a = a.view(B * H, A)  # BxHxA -> (BxH)xA
            mask = mask.view(B * H, -1)
            r = r.view(B * H, -1)  # BxH -> (BxH)x1
            aux_r = aux_r.view(B * H, -1)
            # not using the aux reward
            # r = r + aux_r
            r = r + self.aux_weight * aux_r
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
        self.optimize(s1, a1, prob, r1, aux_r1, s2, done)

        
    def optimize(self, s1, a1, prob, r1, aux_r1, s2, done):
        r = r1 + self.aux_weight * aux_r1
        s_feature = self.state_features.forward(torch.tensor(s1).view(-1).float())
        s_feature_next = self.state_features.forward(torch.tensor(s2).view(-1).float())
        
        # optimize critic
        val_pred = self.critic.forward(s_feature)
        val_next = self.critic.forward(s_feature_next).detach()
        val_exp = r + self.config.gamma * val_next * (1 - done)
        loss_critic = F.mse_loss(val_pred, val_exp)

        # optimize actor

        log_pi, _ = self.actor.get_logprob_dist(s_feature, torch.tensor(a1).view(-1).float())                   # (BxH)xd, (BxH)xA -> (BxH)x1
        td_error = (val_exp - val_pred).detach()    
        loss_actor = -log_pi * td_error

        loss = loss_actor + loss_critic
        self.step(loss)
        # wandb.log({"loss_actor": loss_actor.item(), "loss_critic": loss_critic.item()})
    

        


