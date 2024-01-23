from collections import deque, namedtuple
import numpy as np
import torch
import random
import pickle
transition = namedtuple('transition', 'x, a, r, r_aux, xp, done')

class TrajectoryBuffer:
    """
    Pre-allocated memory interface for storing and using Off-policy trajectories
    Note: slight abuse of notation.
          sometimes Code treats 'dist' as extra variable and uses it to store other things, like: prob, etc.
    """

    def __init__(self, buffer_size, state_dim, action_dim, max_horizon, dist_dim=1, stype=torch.float32,atype = torch.long):

        # Ensure that buffer is strictly longer than max_horizon.
        # This additional space acts like placeholder for s_\infty (which is useful in some algos).
        # Note that mask is off for s_\infty

        # max_horizon = config.env.max_horizon + 1
        max_horizon = max_horizon + 1
        device = 'cpu'

        self.s = torch.zeros((buffer_size, max_horizon, state_dim), dtype=stype, requires_grad=False,
                             device=device)
        self.a = torch.zeros((buffer_size, max_horizon, action_dim), dtype=atype, requires_grad=False,
                             device=device)
        self.beta = torch.ones((buffer_size, max_horizon), dtype=torch.float32, requires_grad=False, device=device)
        self.mask = torch.zeros((buffer_size, max_horizon), dtype=torch.float32, requires_grad=False, device=device)
        self.r = torch.zeros((buffer_size, max_horizon), dtype=torch.float32, requires_grad=False, device=device)
        self.aux_r = torch.zeros((buffer_size, max_horizon), dtype=torch.float32, requires_grad=False, device=device)

        self.buffer_size = buffer_size
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

        self.atype = atype
        self.stype = stype
        # self.config = config

    @property
    def size(self):
        return self.valid_len

    def reset(self):
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

    def next(self):
        self.episode_ctr += 1
        self.buffer_pos += 1

        # Cycle around to the start of buffer (FIFO)
        if self.buffer_pos >= self.buffer_size:
            self.buffer_pos = 0

        if self.valid_len < self.buffer_size:
            self.valid_len += 1

        self.timestep_ctr = 0

        # Fill rewards vector with zeros because episode overwriting this index
        # might have shorter horizon than the previous episode cached in this vector.
        self.r[self.buffer_pos].fill_(0)
        self.aux_r[self.buffer_pos].fill_(0)
        self.mask[self.buffer_pos].fill_(0)

    def add(self, s1, a1, beta1, r1, aux_r1):
        pos = self.buffer_pos
        step = self.timestep_ctr

        self.s[pos][step] = torch.tensor(s1, dtype=self.stype)
        self.a[pos][step] = torch.tensor(a1, dtype=self.atype)
        self.beta[pos][step] = torch.tensor(beta1, dtype=torch.float32)
        self.r[pos][step] = torch.tensor(r1, dtype=torch.float32)
        self.aux_r[pos][step] = torch.tensor(aux_r1, dtype=torch.float32)
        self.mask[pos][step] = torch.tensor(1.0, dtype=torch.float32)

        self.timestep_ctr += 1

    def _get(self, idx):
        return self.s[idx], self.a[idx], self.beta[idx], self.r[idx], self.aux_r[idx], self.mask[idx]

    def sample(self, batch_size):
        count = min(batch_size, self.valid_len)
        return self._get(np.random.choice(self.valid_len, count))

    def sample_last(self, batch_size):
        count = min(batch_size, self.valid_len)
        #FIXME
        idxs = np.array(range(- (count-1), 1)) # @  corrected the numbering by adding (count + 1) instead of count
        idxs = self.buffer_pos + idxs
        return self._get(idxs)

    def sample_wo_last(self, batch_size, delta_size):
        # Sample randomly from the buffer
        # Excluding the last delta trajectories

        count = min(batch_size, self.valid_len - delta_size)
        idxs = np.random.choice(self.valid_len - delta_size, count)
        idxs = self.buffer_pos - delta_size - idxs
        return self._get(idxs)

    def get_all(self):
        return self._get(np.arange(self.valid_len))

    def batch_sample(self, batch_size, randomize=True):
        raise NotImplementedError

    def save(self, path, name):
        dict = {
            's': self.s,
            'a': self.a,
            'beta': self.beta,
            'mask': self.mask,
            'r': self.r,
            'aux_r': self.aux_r,
            'time': self.timestep_ctr, 'pos': self.buffer_pos, 'val': self.valid_len
        }
        with open(path + name + '.pkl', 'wb') as f:
            pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path, name):
        with open(path + name + '.pkl', 'rb') as f:
            dict = pickle.load(f)

        self.s = dict['s']
        self.a = dict['a']
        self.beta = dict['beta']
        self.mask = dict['mask']
        self.r = dict['r']
        self.aux_r = dict['aux_r']
        self.timestep_ctr, self.buffer_pos, self.valid_len = dict['time'], dict['pos'], dict['val']

        print('Memory buffer loaded..')

class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def random_sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_last(self):
        if len(self.buffer) == 0 :
            raise Exception("Buffer Empty")

        return self.buffer[self.location - 1]

    def sample_n_last(self,n):
        if len(self.buffer) < n:
            raise Exception("Buffer smaller")
        samples = []
        for i in range(n):
            samples.append(self.buffer[self.location - 1 -i ])
        return samples



class ReplayBuffer():
    def __init__(self, size=1e5, device=None):

        self.memory = deque(maxlen=int(size))

    def add(self, states, actions, rewards, next_states, dones):

        for i in range(len(states)):
            self.memory.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    def sample(self, sample_size=32):\

        if sample_size > len(self.memory):
            sample_size = len(self.memory)

        samples = random.sample(list(self.memory), sample_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for s in samples:
            states.append(s[0])
            actions.append(s[1])
            rewards.append(s[2])
            next_states.append(s[3])
            dones.append(s[4])

        return states, actions, rewards, next_states, dones


