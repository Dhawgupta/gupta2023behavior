import numpy as np
import torch
from torch import tensor, float32, autograd
from src.algorithms.Agent import Agent
from src.utils import Basis, Critic, Policy, utils, Reward
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
        self.critic = Critic.Critic(state_dim=self.state_features.feature_dim, config=config)
        self.align_rewards = Reward.Alignment_Reward(state_dim=self.state_features.feature_dim, config=config)
        self.align_gamma = Reward.Alignment_Gamma(state_dim=self.state_features.feature_dim, config=config)

        self.memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)

        self.modules = [('actor', self.actor), ('state_features', self.state_features), 
                        ('critic', self.critic), ('rewards', self.align_rewards), ('gamma', self.align_gamma)]


        if self.config.approx == 'First-order':
            self.outer = self.outer_first_order
        
        else:
            raise ValueError("No such approximation technique")
 
        self.counter = 0
        self.inner_mul = 5      # Delta times this is the number of inner optim steps
        self.first_mul = 150    # For first inner optim, these many additional steps are taken
        self.hess_mul = 15      # Batch_size Multiplier when computing Hessian approx
        self.first_time = True
        CRITIC = False
        
     
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

    def update(self, s1, a1, prob, r1, aux_r1, s2, done):
        # Batch episode history
        self.memory.add(s1, a1, prob, r1, aux_r1)
        # self.memory.add(s1, a1, prob, self.gamma_t * r1, aux_r1)
        # self.gamma_t *= self.config.gamma

        # adding second part of the statement to collect atleast first_steps + inner steps data for initial optimization
        # if done and self.counter % self.config.delta == 0 and self.counter > ( self.first_mul + self.inner_mul):
        if done and self.counter % self.config.delta == 0 :
        
            self.optimize()
            # self.memory.reset()         # Throw away all the past data after optimization is done

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



        if self.first_time:
            # Inner optimization loop for delta times to update the policy
            # for iter in range(self.config.delta*self.inner_mul + self.first_time*self.first_mul):
            s, a, beta, r, aux_r, mask = self.memory.sample_last(1)  # BxHxD, BxHxA, BxH, BxH, BxH, BxH
            # use r_p + r_a, to be similar to reinforce
            negJ_phi = - self.get_policy_loss(s, a, beta, r, aux_r, mask, alignment=True,
                                            detach_r=True, actor_reg=self.config.actor_reg, naive = True) # Returns the negative of J_phi
            # self.step(loss) # Ignore this: This updates multiple modules together
            # Compute the total derivative and update the parameters.
            # self.clear_gradients()    # Avoid clearing all gradients in order to track them
            self.actor.optim.zero_grad()

            negJ_phi.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)

            self.actor.step()  # Optimizes in the oppostive direction \theta_{i+1} <- \theta_i - \alpha \delta_\theta (neg J_phi)


        # Flag to avoid the outer optimization loop for the very first call to optimize()
        if self.counter > self.first_mul + self.inner_mul:
            # do self.first_mul + self.inner_mul more updates using the actual r_phi
            for iter in range(self.config.delta * self.inner_mul + self.first_time * self.first_mul):
                s, a, beta, r, aux_r, mask = self.memory.sample(
                    batch_size=batch_size)  # BxHxD, BxHxA, BxH, BxH, BxH, BxH
                negJ_phi = - self.get_policy_loss(s, a, beta, r, aux_r, mask, alignment=True,
                                                  detach_r=True, actor_reg=self.config.actor_reg, naive=False)

                self.actor.optim.zero_grad()
                negJ_phi.backward()  # \delta_\theta (- J_phi)
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)

                self.actor.step()  # \theta_{i+1} <- \theta_i - \alpha \delta_\theta (neg J_phi)
            self.first_time = False



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
            r = self.align_rewards.forward(s_feature, r, aux_r)
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

        # print(s.size(), a.size(), r.size(), aux_r.size(), mask.size())

        if alignment:
            # Obtain alignment rewards and gamma
            aux_r = aux_r.view(B * H, -1)                                   # BxH -> (BxH)x1
            if not naive:
                r = self.align_rewards.forward(s_feature, r, aux_r)       # BHxd, BHx1, BHx1 -> BHx1
            else:
                r = r + aux_r
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
            # gamma_weights[H-i-1] = gamma * gamma_weights[H-i-2]     # gamma_weights to get exact policy grad. 

        # Get action probabilities
        log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a)     # (BxH)xd, (BxH)xA -> (BxH)x1
        log_pi = log_pi.view(B, H)                                # (BxH)x1 -> BxH

        # compute policy grad
        # mean is fine here (instead of sum) as trajectry length is fixed (artifically)
        # Masking is needed here as alignment rewards need not be zero for ghost states in buffer.
        log_pi_return = torch.mean(gamma_weights * mask * log_pi * returns, dim=-1, keepdim=True)   # mean(1xH * BxH * BxH) -> Bx1

        # Compute the final loss
        loss = torch.mean(log_pi_return)                            # mean(Bx1) -> 1


        # Discourage very deterministic policies.
        if self.config.actor_reg > 0:
            if self.config.cont_actions:
                # Isotropic Gaussian for each action dim
                # Taking mean ainstead of sum across action dim to keep things normalized
                # Otherwise entropy coeff will need to depend on action dim as well.
                entropy = torch.sum(dist_all.entropy().view(B, H, -1).mean(dim=-1) * mask) / torch.sum(mask)  # (BxH)xA -> BxH
            else:
                log_pi_all = dist_all.view(B, H, -1)
                pi_all = torch.exp(log_pi_all)                                      # (BxH)xA -> BxHxA
                entropy = - torch.sum(torch.sum(pi_all * log_pi_all, dim=-1) * mask) / torch.sum(mask)

            # Maximize performance (loss)
            # if self.counter > 1050:
            #     print("Entropy" ,entropy.item())
            #     print("Log PI",log_pi_all.mean())
                
            loss = loss + actor_reg * entropy                       # Maximize entropy
            # loss = loss - actor_reg * utils.L2_reg(self.actor)    # Minimize L2
            
            # Over here loss == J_p / J_phi

        # WE RETURN J_p / J_phi
        return loss


  
    def outer_first_order(self, batch_size):    

        # A. Get list of (on policy) grad wrt to new policy (wrt primary reward)
        s, a, beta, r, aux_r, mask = self.memory.sample_last(self.config.delta)            # BxHxD, BxHxA, BxH, BxH, BxH

        Jp = self.outer_policy_loss(s, a, beta, r, aux_r, mask, alignment=False, actor_reg=0)
        outer_policy_grad = autograd.grad(Jp, self.actor.parameters(), retain_graph=False) #  \d_\theta  (Jp)
        

        # Multiply with - negative identity (ND) hessian inverse
        for p in outer_policy_grad:
            p.multiply_(-1.0)    #- \delta_\theta Jp = -I * \delta_\theta  Jp #
        
        # B. Get list of (off policy) grad wrt new policy (wrt to alignment rewards)
        # compute the inner_policy grad with a large batch to get better curvature estiamte later

        s, a, beta, r, aux_r, mask = self.memory.sample_wo_last(self.hess_mul * batch_size, self.config.delta)            # BxHxD, BxHxA, BxH, BxH, BxH

        Jphi = self.get_policy_loss(s, a, beta, r, aux_r, mask, alignment=True, detach_r=False, actor_reg=self.config.actor_reg)
        inner_policy_grad = autograd.grad(Jphi, self.actor.parameters(), create_graph=True, retain_graph=True)  # \d_\theta  Jphi

        
        # C. Get grad wrt to reward, gamma 
        # i.e., Vector-Hessian_inv-Jacobian product
        # Here Hessian is approximated as Identity
        # Equivalent to Neumann_loops = 0 in self.outer_Neumann()

        # The following computes
        # dpi_outer * H^-1 * dr(dpi_inner) 
        # => dpi_outer * dr(dpi_inner)
        # => dr(dpi_outer * dpi_inner)

        # In the following, gradient of (inner_policy_grad's dot product with outer_policy_grad)
        # wrt to reward (or gamma) is computed

        # do reward_l2
        reward_L2 = self.config.reward_L2 * utils.L2_reg(self.align_rewards)

        reward_grad = autograd.grad([*inner_policy_grad, reward_L2], self.align_rewards.parameters(), grad_outputs=[*outer_policy_grad, None], retain_graph=True) #   \delta_phi  (\delta_\theta - Jp * \delta_\theta Jphi) = - \delta_\theta Jp \delta_\phi, \theta Jphi
        #=> = - \delta_\theta Jp \delta_{\phi, \theta} Jphi


        # The post multiplication for - () term as in teh equiation (10)
        for p in reward_grad:
            p.multiply_(-1.0)


        # Don't want gamma param to go to zero; want gamma output ot go to zero
        # gamma_L2 = self.config.gamma_L2 * utils.L2_reg(self.align_gamma)
        
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
        # so that the resultant update is in the positive direction of the gradient
        for param, r_grad in zip(self.align_rewards.parameters(), reward_grad):
            param.grad = - r_grad

        for param, g_grad in zip(self.align_gamma.parameters(), gamma_grad):
            param.grad = -  g_grad
            # print("Gamma param", param.item())

        # Let the desired optimizer use the gradients as apropriate for update
        self.align_rewards.step()
        self.align_gamma.step()


