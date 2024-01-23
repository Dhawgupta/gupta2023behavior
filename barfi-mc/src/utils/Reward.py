from src.utils.Policy import Policy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch import tensor, float32
from src.utils.utils import NeuralNet


class Base_Reward(NeuralNet):
    def __init__(self, state_dim, config):
        super(Base_Reward, self).__init__()
        self.config = config
        self.state_dim = state_dim

    def init(self):
        print("Reward fn: ", [(name, param.shape) for name, param in self.named_parameters()])
        self.optim = self.config.optim(self.parameters(), lr=self.config.reward_lr)




class Base_Gamma(NeuralNet):
    def __init__(self, state_dim, config):
        super(Base_Gamma, self).__init__()
        self.config = config
        self.state_dim = state_dim

    def init(self):
        print("Gamma fn: ", [(name, param.shape) for name, param in self.named_parameters()])
        self.optim = self.config.optim(self.parameters(), lr=self.config.gamma_lr)



class Alignment_Reward(Base_Reward):
    def __init__(self, state_dim, config):
        super(Alignment_Reward, self).__init__(state_dim, config)

        self.fc1 = nn.Linear(state_dim, 3)
        # self.fc2 = nn.Linear(16, 3)
        # self.fc1.weight.data.uniform_(-0, 0)  # comment this if making the critic deeper
        # self.fc1.weight.data.uniform_(0.99, 1.1)
        # init bias unit to zero and the other unit to 1
        # self.fc1.weight.data[0].uniform_(0,0)
        # self.fc1.weight.data[2].uniform_(0.549, 0.55)
        self.init()

    def forward(self, x, r, aux_r):
        x = self.fc1(x)
        # print(x)
        # x = torch.tanh(x)
        # x = self.fc2(x)
        # x1 = x[:, 0:1]
        x1 = torch.tanh(x[:, 0:1])
        x2 = x[:, 1:2]
        x3 = torch.tanh(x[:, 2:3])
        # print("Init Values" ,x1.mean(),x3.mean())

        # print('==>',x3.data, aux_r.data)
        # print(x.size(), r.size(), aux_r.size(), x[:,2:3].size(), (x[:,2:3]*aux_r).size())
        # Need to slice second dimensions to retain second axis
        # without second axis, it becomes B * Bx1 -> BxB
        # print("init values", x[:,0:1].mean(), x[:,2:3].mean())
        # we will be using reward regularization to avoid exploding rewards
        return x[:, 0:1] + x[:, 1:2] * r + x[:, 2:3] * aux_r
        # return x1 + r + x3 * aux_r
        # return x[:, 0:1] + r + x3 * aux_r
        # return x[:,0:1] + r + aux_r * x[:, 2:3]
        
        # return x[:,0:1] + r * x[:,1:2] + aux_r * x[:,2:3]
        # print("x1" , x1.mean())
        # print("x3", x[:,2:3].mean())
        # return x1 + r + x[:, 2:3] * aux_r

    def forward_outputs(self, x):
        # just return the forwardded outrput
        x = self.fc1(x)
        # x1 = x[:, 0:1]
        x1 = torch.tanh(x[:, 0:1])
        x2 = x[:, 1:2]
        x3 = torch.tanh(x[:, 2:3])
        # return x1, x3
        # return x[:,0:1], x[:, 1:2], x[:, 2:3]
        with torch.no_grad():
            return torch.mean(torch.abs(self.fc1.weight.data[0])), torch.mean(torch.abs(self.fc1.weight.data[2]))
        # return x[:,0:1], x[:, 2:3]
        # return x[:,0:1], x3

class Alignment_Gamma(Base_Gamma):
    def __init__(self, state_dim, config):
        super(Alignment_Gamma, self).__init__(state_dim, config)

        # self.fc1 = nn.Linear(state_dim, 1)
        self.gamma_param = torch.nn.Parameter(torch.tensor([4.6])) # changed from 5.0 to 4.6 to start at exactly 0.99
        # self.fc1.weight.data.uniform_(-0, 0)  # comment this if making the critic deeper
        self.init()

    def forward(self):
        return torch.sigmoid(self.gamma_param)


