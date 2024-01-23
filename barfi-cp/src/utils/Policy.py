import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, float32
from torch.distributions import Normal
from src.utils.utils import NeuralNet


class Policy(NeuralNet):
    def __init__(self, state_dim, config, action_dim=None):
        super(Policy, self).__init__()

        self.config = config
        self.state_dim = state_dim
        if action_dim is None:
            self.action_dim = config.env.action_space.shape[0]
        else:
            self.action_dim = action_dim

    def init(self):
        temp, param_list = [], []
        for name, param in self.named_parameters():
            temp.append((name, param.shape))
            if 'var' in name:
                param_list.append(
                    {'params': param, 'lr': self.config.actor_lr / 100})  # Keep learning rate of variance much lower
            else:
                param_list.append({'params': param})
        self.optim = self.config.optim(param_list, lr=self.config.actor_lr)

        print("Policy: ", temp)




def get_Policy(state_dim, config):
    if config.cont_actions:
        atype = torch.float32
        actor = Insulin_Gaussian(state_dim=state_dim, config=config)
        action_size = actor.action_dim
    else:
        atype = torch.long
        action_size = 1
        actor = Categorical(state_dim=state_dim, config=config)

    return actor, atype, action_size


class Categorical(Policy):
    def __init__(self, state_dim, config, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)

        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim

        self.random = np.ones(self.action_dim)/self.action_dim

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)
        # self.fc1 = nn.Linear(self.state_dim, 16)
        # self.fc2 = nn.Linear(16, self.action_dim)
        self.init()

    def re_init_optim(self):
        # In case we later want to fine-tune policy on primary rewards, without regularization
        self.optim = self.config.optim(self.parameters(), lr=self.config.actor_lr)

    def forward(self, state):
        x = self.fc1(state)
        
        # x = F.relu(x)

        # x = self.fc2(x)
        
        return x

    def get_action(self, state, explore=0, behavior=False):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        probs = dist.cpu().view(-1).data.numpy()

        rho = 1
        if behavior:
            # Create behavior policy
            # By mixing evaluation policy and random policy
            new_probs = self.config.alpha * probs +  (1-self.config.alpha) * self.random

            # Bug with numpy, floating point errors don't let prob to sum to 1 exactly
            # https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
            # solution is to normalize the probabilities by dividing them by their sum if the sum is close enough to 1
            new_probs /= new_probs.sum()
            action = np.random.choice(self.action_dim, p=new_probs)

            beta = new_probs[action]
            pi = probs[action]
            rho = pi/beta
        else:
            action = np.random.choice(self.action_dim, p=probs)

        return action, rho, dist

    def get_logprob_dist(self, state, action):
        x = self.forward(state)                                                              # BxA
        log_dist = F.log_softmax(x, -1)                                                      # BxA
        return log_dist.gather(1, action), log_dist                                          # BxAx(Bx1) -> B



class Insulin_Gaussian(Policy):
    def __init__(self, state_dim, config):
        super(Insulin_Gaussian, self).__init__(state_dim, config, action_dim=2)

        # Set the ranges or the actions
        self.low, self.high = config.env.action_space.low * 1.0, config.env.action_space.high * 1.0
        self.action_low = tensor(self.low, dtype=float32, requires_grad=False, device=config.device)
        self.action_diff = tensor(self.high - self.low, dtype=float32, requires_grad=False, device=config.device)

        self.random_mean = torch.tensor((self.low + self.high)/2)

        print("Action Low: {} :: Action High: {}".format(self.low, self.high))

        # Initialize network architecture and optimizer
        self.fc_mean = nn.Linear(state_dim, 2)
        self.init()

    def forward(self, state):
        mean = torch.sigmoid(self.fc_mean(state)) * self.action_diff + self.action_low  # BxD -> BxA
        std = torch.ones_like(mean, requires_grad=False) * self.config.gauss_std  # BxD -> BxA
        return mean, std

    def get_action(self, state, explore=0, behavior=False):
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        rho = 1
        if behavior:
            new_mean = self.config.alpha * mean + (1-self.config.alpha) * self.random_mean
            new_dist = Normal(new_mean, std)
            action = new_dist.sample()

            # Pytorch doesn't have a direct function for computing prob, only log_prob.
            # Hence going the round-about way.
            # prob = poduct of all probabilities. Therefore log is the sum of them.
            logbeta = new_dist.log_prob(action).view(-1).data.numpy().sum(axis=-1)
            logpi = dist.log_prob(action).view(-1).data.numpy().sum(axis=-1)


            # DO not do the commented lines; It can be numerically unstable
            # pi = np.exp(logpi)
            # beta = np.exp(logbeta)
            # rho2 = pi/beta

            rho = np.exp(logpi - logbeta)

            # print(action, new_dist.log_prob(action), dist.log_prob(action), logpi, logbeta, rho, rho2)

        else:
            action = dist.sample()


        action = action.cpu().view(-1).data.numpy()
        return action, rho


    def get_logprob_dist(self, state, action):
        mean, var = self.forward(state)                                                         # BxA, BxA
        dist = Normal(mean, var)                                                                # BxAxdist()
        return dist.log_prob(action).sum(dim=-1), dist                                          # BxAx(BxA) -> B
