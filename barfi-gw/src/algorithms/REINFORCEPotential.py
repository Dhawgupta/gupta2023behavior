import numpy as np
import torch
from torch import tensor, float32
from src.algorithms.Agent import Agent
from src.utils import Basis, Critic, Policy, utils
import torch.functional as F

"""
TODO
- Disable actor L2 here
"""


class REINFORCEPotential(Agent):
    def __init__(self, config):
        super(REINFORCEPotential, self).__init__(config)

        # Get state features and instances for Actor, Value, Reward, and Gamma functions
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features.feature_dim,
                                                                     config=config)


        self.memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)

        self.modules = [('actor', self.actor), ('state_features', self.state_features)]

        self.counter = 0
        self.init()

    def reset(self):
        super(REINFORCEPotential, self).reset()
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
            r = r + aux_r
            gamma = self.config.gamma
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

        if done and self.counter % self.config.delta == 0:
            loss = -1.0 * self.optimize()
            self.actor.optim.zero_grad()
            loss.backward()
            self.actor.step()

            # self.memory.reset()  # Throw away all the past data after optimization is done

    def optimize(self):
        # print(self.memry)
        s, a, beta, r, aux_r, mask = self.memory.sample_last(0)  # BxHxD, BxHxA, BxH, BxH, BxH, BxH

        # r = r + aux_r  # Combine both primary and the auxiliary rewards

        B, H, D = s.shape
        _, _, A = a.shape

        # create state features
        s_feature = self.state_features.forward(s.view(B * H, D))  # BxHxD -> (BxH)xd
        a = a.view(B * H, A)  # BxHxA -> (BxH)xA
        r = r.view(B * H, -1)  # BxH -> (BxH)x1
        mask = mask.view(B * H, -1)  # BxH -> (BxH)x1
        r = r * mask
        r = r.view(B, H)
        mask = mask.view(B, H)
        gamma = self.config.gamma
        returns = torch.zeros_like(r)
        returns[:, H - 1] = r[:, H - 1]
        gamma_weights = torch.ones((1, H))
        for i in range(H - 2, -1, -1):
            returns[:, i] = r[:, i] + gamma * returns[:, i + 1].clone()
            # gamma_weights[H-i-1] = gamma * gamma_weights[H-i-2]     # gamma_weights to get exact policy grad.
        returns = returns - aux_r
        # Get action probabilities
        log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a)  # (BxH)xd, (BxH)xA -> (BxH)x1
        log_pi = log_pi.view(B, H)  # (BxH)x1 -> BxH

        # compute policy grad
        # mean is fine here (instead of sum) as trajectry length is fixed (artifically)
        # Masking is needed here as alignment rewards need not be zero for ghost states in buffer.
        log_pi_return = torch.mean(gamma_weights * mask * log_pi * returns, dim=-1,
                                   keepdim=True)  # mean(1xH * BxH * BxH) -> Bx1

        # Compute the final loss
        loss = torch.mean(log_pi_return)  # mean(Bx1) -> 1

        # Discourage very deterministic policies.
        if self.config.actor_reg > 0:
            # print("Doing Entropy")
            if self.config.cont_actions:
                # Isotropic Gaussian for each action dim
                # Taking mean ainstead of sum across action dim to keep things normalized
                # Otherwise entropy coeff will need to depend on action dim as well.
                entropy = torch.sum(dist_all.entropy().view(B, H, -1).mean(dim=-1) * mask) / torch.sum(
                    mask)  # (BxH)xA -> BxH
            else:
                log_pi_all = dist_all.view(B, H, -1)
                pi_all = torch.exp(log_pi_all)  # (BxH)xA -> BxHxA
                entropy = - torch.sum(torch.sum(pi_all * log_pi_all, dim=-1) * mask) / torch.sum(mask)

            # Maximize performance (loss)
            # if self.counter > 1050:
            #     print("Entropy" ,entropy.item())
            #     print("Log PI",log_pi_all.mean())

            loss = loss + self.config.actor_reg * entropy  # Maximize entropy
            # loss = loss - actor_reg * utils.L2_reg(self.actor)    # Minimize L2

        return loss
        # self.clear_gradients()

        # self.step(loss)


