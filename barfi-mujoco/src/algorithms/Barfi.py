import numpy as np
import torch
from torch import tensor, float32, autograd
from src.algorithms.Agent import Agent
from src.utils import Basis, Policy, utils, Reward
import torch.functional as F

# torch.autograd.set_detect_anomaly(True)

"""
TODO
- [IMP] Check the inner/outer grad sign requirement during implicit-grad omputation
- [IMP] Try entropy regularization instead of L2
- [IMP] Try with tabular
- [IMP] Visualize the axuliary rewards learnt
- [IMP] Understand the effect of different optimizers for example, ADAM or SGD in terms of past stored information on gradients

- [IMP] PLay around with the architecture for the reward/gamma module 
- This is for a fixed basis function
    > Need to have separate basis functions for inner and outer loop when basis is shared
- Decide whether gamma-discounted or gamma-dropped policy-grad should be used for outer-loop

- Initialization of Gamma has big impacts on performance
- Actor L2 matters for hard problems
"""

class Barfi(Agent):
    def __init__(self, config):
        super(Barfi, self).__init__(config)

        # Get state features and instances for Actor, Value, Reward, and Gamma functions
        #
        # Note:
        # Algin_Returns is a function only of state
        # Align_Gamma is a scalar

        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features.feature_dim,
                                                                       config=config)
        self.align_rewards = Reward.Alignment_Reward(state_dim=self.state_features.feature_dim, config=config)
        self.align_gamma = Reward.Alignment_Gamma(state_dim=self.state_features.feature_dim, config=config)
        # self.normalize_gamma = config.normalize_gamma

        self.memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)

        self.modules = [('actor', self.actor), ('state_features', self.state_features), ('rewards', self.align_rewards), ('gamma', self.align_gamma)]

        self.outer = self.outer_first_order
        self.aux_weight = config.aux_weight
        self.running_return_aux = None
        self.running_return_outer = None
        
        self.counter = 0
        # self.inner_mul = 5      # Delta times this is the number of inner optim steps
        self.inner_mul = config.inner_mul
        # self.first_mul = 150    # For first inner optim, these many additional steps are taken
        self.first_mul = config.first_mul
        self.hess_mul = 15      # Batch_size Multiplier when computing Hessian approx
        self.first_time = True
        
        self.outer_policy_loss = self.get_policy_loss
        
        self.init()

    def reset(self):
        super(Barfi, self).reset()
        self.memory.next()
        self.counter += 1
        self.gamma_t = 1

    def get_action(self, state, explore=0):
        explore = 0  # Don't do eps-greedy with policy gradients
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
        action, prob, dist = self.actor.get_action(state, explore=explore)

        if self.config.debug:
            self.track_entropy(dist, action)

        # print(action, dist)
        return action, prob

    def get_gamma_normalizer(self, gamma, episode_length):
        gamma_norm = 0
        gamma_temp = 1
        for i in range(episode_length-1):
            gamma_norm += (i+1) * (gamma_temp)
            gamma_temp *= gamma
        return gamma_norm

    def get_gamma_normalizer_reward(self, gamma, rewards, episode_length):
        gamma_norm = 0
        gamma_temp = 1
        for i in range(episode_length-1):
            gamma_norm += (i+1) * (gamma_temp) * rewards[i+1]
            gamma_temp *= gamma
    
        return gamma_norm


    def update(self, s1, a1, prob, r1, aux_r1, s2, done):
        # Batch episode history
        self.memory.add(s1, a1, prob, r1, aux_r1)
    
        if done and self.counter % self.config.delta == 0 and self.counter > ( self.first_mul + self.inner_mul):
            self.optimize()
            # self.memory.reset()         # Throw away all the past data after optimization is done

        # adding second part of the statement to collect atleast first_steps + inner steps data for initial optimization
        # if done and self.counter % self.config.delta == 0 and self.counter > ( self.first_mul + self.inner_mul):
        # if done and self.counter % self.config.delta == 0 :
        #     self.optimize()
        #     self.memory.reset()         # Throw away all the past data after optimization is done

    def optimize(self):
        batch_size = self.memory.size if self.memory.size < self.config.batch_size else self.config.batch_size

        # Update the reward and gamma
        # changing the order of reward update and policy update
        # such that sampling occurs between the reward/gamma update and policy update

        # Outer optimization loop
        if not self.first_time:
            self.clear_gradients()
            reward_grads, gamma_grads = self.outer(batch_size)
            self.update_reward_gamma(reward_grads, gamma_grads)


        for iter in range(self.config.delta*self.inner_mul + self.first_time*self.first_mul):
            print(f"Updates : {iter}")
            s, a, beta, r, aux_r, mask = self.memory.sample(batch_size)
            loss = - 1.0 * self.get_policy_loss(s, a, beta, r, aux_r, mask, alignment=True,detach_r=True, actor_reg=self.config.actor_reg, naive = False)
            # self.actor.optim.zero_grad()
            self.actor.zero_grad()
            # self.state_features.zero_grad()
            self.state_features.zero_grad()
            loss.backward()
            self.actor.step()
            self.state_features.step()
            # self.actor.optim.step()
            # self.state_features.step()
            # self.actor.optim.zero_grad()
            # self.state_features.optim.zero_grad()

        self.first_time = False
        return loss


    def get_aligned_reward_terms(self, s):
        with torch.no_grad():
            # s = torch.tensor(s)
            B, D = s.shape
            s_feature = self.state_features.forward(s.view(B, D))
            x1, x3 = self.align_rewards.forward_outputs(s_feature)
            return x1.view(B).numpy(), x3.view(B).numpy()

    def get_aligned_return_gamma(self):
        # return the aligned return 
        with torch.no_grad():
            s, a, beta, r, aux_r, mask = self.memory.sample_last(0)
            B, H, D = s.shape
            _, _, A = a.shape

            # create state features
            s_feature = self.state_features.forward(s.view(B * H, D))           # BxHxD -> (BxH)xd
            a = a.view(B * H, A)                                                # BxHxA -> (BxH)xA
            r = r.view(B * H, -1)                                               # BxH -> (BxH)x1
            mask = mask.view(B*H, -1)
            aux_r = aux_r.view(B * H, -1)
            r = self.align_rewards.forward(s_feature, r, self.aux_weight * aux_r)
            gamma = self.align_gamma.forward()
            r = r * mask
            r = r.view(B, H)
            # mask = mask.view(B, H)
            returns = torch.zeros_like(r)
            returns[:, H-1] = r[:, H-1]
            gamma_weights = torch.ones((1, H)) 
            for i in range(H-2, -1, -1):
                returns[:, i] = r[:, i] + gamma * returns[:, i+1].clone()
            index_end = torch.argmin(mask) - 1
            return torch.mean(returns[0,:index_end+1]).item() , gamma.item()



    def get_policy_loss(self, s, a, beta, r, aux_r, mask, alignment=True, detach_r=True, actor_reg=0, naive = False):
        """
        Computes the grads and stores them in the grad variable of the actors parameters
        """
        B, H, D = s.shape
        _, _, A = a.shape

        # create state features
        s_feature = self.state_features.forward(s.view(B * H, D))           # BxHxD -> (BxH)xd
        a = a.view(B * H, A)                                                # BxHxA -> (BxH)xA
        r = r.view(B * H, -1)                                               # BxH -> (BxH)x1
        mask = mask.view(B*H, -1)

        if alignment:
            aux_r = aux_r.view(B * H, -1)                                   # BxH -> (BxH)x1
            if not naive:
                r = self.align_rewards.forward(s_feature, r, self.aux_weight *  aux_r)       # BHxd, BHx1, BHx1 -> BHx1
            else:
                r = r + self.aux_weight * aux_r
            gamma = self.align_gamma.forward()

            if detach_r:
                # Backprop through r and gamma not needed
                r = r.detach()
                gamma = gamma.detach()
            else:
                # print("Gamma:{:.2f} :: Mean abs reward {:.2f} :: Max abs reward {:.2f}".format(gamma.item(), torch.mean(torch.abs(r)), torch.max(torch.abs(r))))
                pass
        else:
            # Use only the primary rewards
            # returns = r.clone()             # Returns are changed in-place later; so clone r.
            gamma = self.config.gamma


        # Reverse sum all the returns to get sum of discounted future returns
        # mask the rewards appropriately for return calcualation
        r = r * mask
        r = r.view(B, H)
        mask = mask.view(B, H)
        returns = torch.zeros_like(r)
        returns[:, H-1] = r[:, H-1]
        gamma_weights = torch.ones((1, H)) 
        for i in range(H-2, -1, -1):
            returns[:, i] = r[:, i] + gamma * returns[:, i+1].clone()
        
        if alignment:
            if self.running_return_aux is not None:
                return_norm = (returns - self.running_return_aux) * mask
            else:
                return_norm = returns * mask
        else:
            if self.running_return_outer is not None:
                return_norm = (returns - self.running_return_outer) * mask
            else:
                return_norm = returns * mask

        with torch.no_grad():
            if alignment:
                if self.running_return_aux is None:
                    self.running_return_aux = torch.mean(returns).item()
                else:
                    self.running_return_aux = 0.9 * self.running_return_aux + 0.1 * torch.mean(returns).item()
                
            else:
                if self.running_return_outer is None:
                    self.running_return_outer = torch.mean(returns).item()
                else:
                    self.running_return_outer = 0.99 * self.running_return_outer + 0.01 * torch.mean(returns).item()
            
        # Get action probabilities
        log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a)     # (BxH)xd, (BxH)xA -> (BxH)x1
        log_pi = log_pi.view(B, H)                                # (BxH)x1 -> BxH

        # compute policy grad
        # mean is fine here (instead of sum) as trajectry length is fixed (artifically)
        # Masking is needed here as alignment rewards need not be zero for ghost states in buffer.
        log_pi_return = torch.mean(gamma_weights * mask * log_pi * return_norm, dim=-1, keepdim=True)   # mean(1xH * BxH * BxH) -> Bx1

        # Compute the final loss
        loss = torch.mean(log_pi_return)                            # mean(Bx1) -> 1

    
        return loss


  
    def outer_first_order(self, batch_size):    

        # A. Get list of (on policy) grad wrt to new policy (wrt primary reward)
        s, a, beta, r, aux_r, mask = self.memory.sample_last(self.config.delta)            # BxHxD, BxHxA, BxH, BxH, BxH

        Jp = self.outer_policy_loss(s, a, beta, r, aux_r, mask, alignment=False, actor_reg=0)
        outer_policy_grad = autograd.grad(Jp, self.actor.parameters(), retain_graph=False) #  \d_\theta  (Jp)
        

        # Multiply with - negative identity (ND) hessian inverse
        # The Below should not be there I think
        for p in outer_policy_grad:
            p.multiply_(-1.0)    #- \delta_\theta Jp = -I * \delta_\theta  Jp #
        
        # B. Get list of (off policy) grad wrt new policy (wrt to alignment rewards)
        # compute the inner_policy grad with a large batch to get better curvature estiamte later

        s, a, beta, r, aux_r, mask = self.memory.sample_wo_last(self.hess_mul * batch_size, self.config.delta)            # BxHxD, BxHxA, BxH, BxH, BxH

        Jphi = self.get_policy_loss(s, a, beta, r, aux_r, mask, alignment=True, detach_r=False, actor_reg=self.config.actor_reg)
        inner_policy_grad = autograd.grad(Jphi, self.actor.parameters(), create_graph=True, retain_graph=True)  # \d_\theta  Jphi

    
        # do reward_l2
        reward_L2 = self.config.reward_L2 * utils.L2_reg(self.align_rewards)

        reward_grad = autograd.grad([*inner_policy_grad, reward_L2], self.align_rewards.parameters(), grad_outputs=[*outer_policy_grad, None], retain_graph=True) #   \delta_phi  (\delta_\theta - Jp * \delta_\theta Jphi) = - \delta_\theta Jp \delta_\phi, \theta Jphi
        #=> = - \delta_\theta Jp \delta_{\phi, \theta} Jphi


        # The post multiplication for - () term as in teh equiation (10)
        for p in reward_grad:
            p.multiply_(-1.0)


        # Don't want gamma param to go to zero; want gamma output ot go to zero
        # gamma_L2 = self.config.gamma_L2 * utils.L2_reg(self.align_gamma)
        # if self.normalize_gamma:
    
        #     # print("Normalizing gamma")
        #     # print(inner_policy_grad)
        #     with torch.no_grad():
        #         gamma_temp = self.align_gamma.forward()
        #         gamma_norm = self.get_gamma_normalizer(gamma_temp, mask.shape[1])
        #     inner_policy_grad = [p / gamma_norm for p in inner_policy_grad]
        #     # print(inner_policy_grad)


        gamma_L2 = self.config.gamma_L2 * self.align_gamma.forward()  # This we use as positive, because the optimizer will take negative direction for this
        gamma_grad = autograd.grad([*inner_policy_grad, gamma_L2], self.align_gamma.parameters(), grad_outputs=[*outer_policy_grad, None])

        for p in gamma_grad:
            p.multiply_(-1.0) # doing the post multiplication of gradient before, becauyse of combinatio of l2 grad


        return reward_grad, gamma_grad


    def update_reward_gamma(self, reward_grad, gamma_grad):
        # Set the respective reward and gamma gradients in the 
        # d_p (derivative of parameters) placeholder
        #
        # Note: Optimizer will take step in negative direction of the gradient
        # But we want to maximize the objective, 
        # Therefore, grad stores the negative gradient
        # so that the resultant update is in the positive direction of the gradient\
        self.align_gamma.zero_grad()
        self.align_rewards.zero_grad()
        for param, r_grad in zip(self.align_rewards.parameters(), reward_grad):
            param.grad = - r_grad

        for param, g_grad in zip(self.align_gamma.parameters(), gamma_grad):
            param.grad = -  g_grad
            # print("Gamma param", param.item())

        # Let the desired optimizer use the gradients as apropriate for update
        self.align_rewards.step()
        self.align_gamma.step()



