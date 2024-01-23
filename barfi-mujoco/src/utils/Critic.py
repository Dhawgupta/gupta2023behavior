import numpy as np
import torch
from torch import float32
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from src.utils.utils import NeuralNet


class Base_Critic(NeuralNet):
    def __init__(self, state_dim, config):
        super(Base_Critic, self).__init__()
        self.config = config
        self.state_dim = state_dim

    def init(self):
        print("Critic: ", [(name, param.shape) for name, param in self.named_parameters()])
        # self.optim = self.config.optim(self.parameters(), lr=self.config.critic_lr)
        # use the same learning rate as actor
        self.optim = self.config.optim(self.parameters(), lr=(self.config.actor_lr * self.config.critic_lr_ratio))  
        



class Critic(Base_Critic):
    def __init__(self, state_dim, config):
        super(Critic, self).__init__(state_dim, config)

        self.fc1 = nn.Linear(state_dim, 1)
        # self.fc1 = nn.Linear(self.state_dim, 16)
        # self.fc2 = nn.Linear(16, 1)
        # self.fc1.weight.data.uniform_(-0, 0)  # comment this if making the critic deeper
        self.init()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        return x


