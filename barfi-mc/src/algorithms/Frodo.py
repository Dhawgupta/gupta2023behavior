import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import tensor, float32, autograd
from src.algorithms.Agent import Agent
from src.utils import Basis, Policy, utils, Reward
import torch.functional as F
import higher

class Frodo(Agent):
    def __init__(self, config):
        super(Frodo, self).__init__(config)
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features.feature_dim,
                                                                       config=config)
        self.align_rewards = Reward.Alignment_Reward(state_dim=self.state_features.feature_dim, config=config)
        # self.align_gamma = Reward.Alignment_Gamma(state_dim=self.state_features.feature_dim, config=config)
        # self.normalize_gamma = config.normalize_gamma

        self.memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)

        # self.modules = [('actor', self.actor), ('state_features', self.state_features), ('rewards', self.align_rewards), ('gamma', self.align_gamma)]
        self.modules = [('actor', self.actor), ('rewards', self.align_rewards)]
        self.actor_optim = self.actor.optim
        # self.state_optim = self.state_features.optim
        self.reward_optim = self.align_rewards.optim
        
        self.running_return_aux = None
        self.running_return_outer = None
        
        self.counter = 0
        self.inner_mul = config.inner_mul
        
        
        self.init()
    
    def reset(self):
        super(Frodo, self).reset()
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
    
        if done and self.counter % self.config.delta == 0 and self.counter > ( self.inner_mul):
            self.optimize()
            # self.memory.reset()         # Throw away all the past data after optimization is done

        # adding second part of the statement to collect atleast first_steps + inner steps data for initial optimization
        # if done and self.counter % self.config.delta == 0 and self.counter > ( self.first_mul + self.inner_mul):
        # if done and self.counter % self.config.delta == 0 :
        #     self.optimize()
        #     self.memory.reset()         # Throw away all the past data after optimization is done


    def optimize(self):
        batch_size = self.memory.size if self.memory.size < self.config.batch_size else self.config.batch_size

        def grad_callback(grads):
            grad_new = []
            # print(grads)
            with torch.no_grad():
                flat_tensor = [g.view(-1) for g in grads]
                concat_tensor = torch.cat(flat_tensor, dim=0)
                norm = torch.norm(concat_tensor)
                num_ele = sum(t.numel() for t in grads)
                # Commented out the below, because this is the factor
                # norm /= torch.sqrt(torch.tensor(num_ele, dtype=torch.float))
                # print(norm)
            if norm > 1.0:
                for p in grads:
                    # shape = p.shape
                    grad_new.append(p / norm)
            else:
                grad_new = grads
            
            # with torch.no_grad():
            #     flat_tensor = [g.view(-1) for g in grad_new]
            #     concat_tensor = torch.cat(flat_tensor, dim=0)
            #     norm = torch.norm(concat_tensor)
            #     num_ele = sum(t.numel() for t in grad_new)
            #     norm /= torch.sqrt(torch.tensor(num_ele, dtype=torch.float))
            #     print(norm)
            return tuple(grad_new)
            

            # for p in grads:
            #     shape = p.shape
            #     # grad_new.append(torch.nn.functional.normalize(p, dim=0))
            #     # calculate norm of a vector
            #     with torch.no_grad():
            #         norm = torch.norm(p.reshape(-1))
            #     # print(norm)
            #     if norm > 1.0:
            #         grad_new.append(torch.nn.functional.normalize(p.reshape(-1), dim=0).reshape(shape))
            #     else:
            #         grad_new.append(p)
            #     # grad_new.append(torch.nn.functional.normalize(p.reshape(-1), dim=0).reshape(shape))
            # # print(grad_new)
            # return tuple(grad_new)

            # normalize the vectors
            # return torch.nn.functional.normalize(grads)
            # torch.nn.utils.clip_grad_norm_(params, 1)
        # start with innner optimization for some steps
        with higher.innerloop_ctx(self.actor, self.actor_optim) as (actormodel, actoropt):
            # with higher.innerloop_ctx(self.state_features, self.state_optim) as (state_featuresmodel, state_featuresopt):
            for iter in range(self.config.delta * self.inner_mul):
                print(f"Updates : {iter}")
                # for p in actormodel.parameters():
                #     print(torch.mean(p))
                # for p in state_featuresmodel.parameters():
                #     print(torch.mean(p))

                s, a, beta, r, aux_r, mask = self.memory.sample(batch_size)
                loss = - 1.0 * self.get_policy_loss(s, a, beta, r, aux_r, mask, alignment=True,detach_r=False, actor_reg=self.config.actor_reg, naive = False, actor=actormodel, state_features=self.state_features)
                actoropt.step(loss, grad_callback=grad_callback)
                # state_featuresopt.step(loss, grad_callback=grad_callback)


            # do  outer optimization for some ste
            # calculate outer loss
            s, a, beta, r, aux_r, mask = self.memory.sample_last(self.config.delta)
            loss_outer = -1.0 * self.get_policy_loss(s, a, beta, r, aux_r, mask, alignment=False, detach_r=True, actor_reg=0.0, actor=actormodel, state_features=self.state_features)

            # Outer optimization loop
            self.reward_optim.zero_grad()
            loss_outer.backward()
            # for p in self.align_rewards.parameters():
            #     print(p.grad)
                # p.grad.data.clamp_(-1, 1)

            #NOTE diagnostic step to check if gradient clipping is working
            # grads = []
            # for p in self.align_rewards.parameters():
            #     grads.append(p.grad)
            # print("Old", grads)
            # new_grads = grad_callback(grads)
            # print("Self",new_grads)
            # print(list(self.align_rewards.parameters())[0].grad)
            torch.nn.utils.clip_grad_norm_(self.align_rewards.parameters(), 1)
            # grads = []
            # for p in self.align_rewards.parameters():
            #     grads.append(p.grad)
            # print("Method",grads)
            self.reward_optim.step()

            with torch.no_grad():
                for pm, pt in zip(self.actor.parameters(), actormodel.parameters()):
                    pm.data = pt.data
                # for pm, pt in zip(self.state_features.parameters(), state_featuresmodel.parameters()):
                #     pm.data = pt.data
        
                
        return loss_outer.item()
    

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
            r = self.align_rewards.forward(s_feature, r,  aux_r)
            # gamma = self.align_gamma.forward()
            gamma = self.config.gamma
            r = r * mask
            r = r.view(B, H)
            # mask = mask.view(B, H)
            returns = torch.zeros_like(r)
            returns[:, H-1] = r[:, H-1]
            gamma_weights = torch.ones((1, H)) 
            for i in range(H-2, -1, -1):
                returns[:, i] = r[:, i] + gamma * returns[:, i+1].clone()
            index_end = torch.argmin(mask) - 1
            return torch.mean(returns[0,:index_end+1]).item() , gamma



    def get_policy_loss(self, s, a, beta, r, aux_r, mask, alignment=True, detach_r=True, actor_reg=0, naive = False, actor = None, state_features = None, align_rewards = None, align_gamma = None):
        """
        Computes the grads and stores them in the grad variable of the actors parameters
        """
        if actor is None:
            actor = self.actor
        if state_features is None:
            state_features = self.state_features
        if align_rewards is None:
            align_rewards = self.align_rewards
        if align_gamma is None:
            # align_gamma = self.align_gamma
            align_gamma = self.config.gamma
        
        B, H, D = s.shape
        _, _, A = a.shape

        # create state features
        s_feature = state_features.forward(s.view(B * H, D))           # BxHxD -> (BxH)xd
        a = a.view(B * H, A)                                                # BxHxA -> (BxH)xA
        r = r.view(B * H, -1)                                               # BxH -> (BxH)x1
        mask = mask.view(B*H, -1)

        if alignment:
            aux_r = aux_r.view(B * H, -1)                                   # BxH -> (BxH)x1
            if not naive:
                r = align_rewards.forward(s_feature, r,   aux_r)       # BHxd, BHx1, BHx1 -> BHx1
            else:
                r = r +  aux_r
            # gamma = align_gamma.forward()
            gamma = self.config.gamma

            if detach_r:
                # Backprop through r and gamma not needed
                r = r.detach()
                # gamma = gamma.detach()
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
        log_pi, dist_all = actor.get_logprob_dist(s_feature, a)     # (BxH)xd, (BxH)xA -> (BxH)x1
        log_pi = log_pi.view(B, H)                                # (BxH)x1 -> BxH

        # compute policy grad
        # mean is fine here (instead of sum) as trajectry length is fixed (artifically)
        # Masking is needed here as alignment rewards need not be zero for ghost states in buffer.
        log_pi_return = torch.mean(gamma_weights * mask * log_pi * return_norm, dim=-1, keepdim=True)   # mean(1xH * BxH * BxH) -> Bx1

        # Compute the final loss
        loss = torch.mean(log_pi_return)                            # mean(Bx1) -> 1

    
        return loss
