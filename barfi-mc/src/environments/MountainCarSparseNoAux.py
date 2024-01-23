'''
Dense reward implementation of MC, where the reward is 0  at every time step and 1 at the end
'''
import sys , os, time
sys.path.append(os.getcwd())
import numpy as np
from src.utils.utils import Space


class MountainCarSparseNoAux(object):
    def __init__(self, action_type = 'discrete', n_actions = 3):
        self.action_space = Space(size = n_actions)
        self.observation_space = Space(low = np.array([-1.2, -0.07]), high=  np.array([0.5, 0.07]), dtype = np.float32)
        self.episode = 0
        self.max_horizon = 2000
        self.action_repetition = 1
        self.aux_vel_scale = 10
        self.n_actions = n_actions
        self.reset()

    def seed(self, seed):
        self.seed = seed
    
    def reset(self):
        self.episode += 1
        self.steps_taken  = 0
        self.position = -0.6 + np.random.random() * 0.2
        self.velocity = 0
        self.curr_state = np.array([self.position, self.velocity])
        return self.curr_state.copy()
    
    def step(self, action):

        self.steps_taken += 1
        a = action - 1
        reward = 0
        aux_reward = 0

        term = self.is_terminal()
        # print(term)
        if term:
            return self.curr_state, self.get_reward(), 0 , term, {'No Info Implemented yet'}
        for rep in range(self.action_repetition):
            self.velocity += 0.001 * a - 0.0025  * np.cos(3 * self.position)
            if self.velocity < - 0.07:
                self.velocity = -0.07
            elif self.velocity >= 0.07:
                self.velocity = 0.069999999
            self.position += self.velocity
            if self.position < -1.2:
                self.position = -1.2
                self.velocity = 0.0
            self.curr_state = np.array([self.position, self.velocity])
            aux_reward += self.get_aux_reward(action = action)
            term = self.is_terminal()
            reward += self.get_reward()
            # print(reward)
            if term:
                break
        # print(self.steps_taken)
        # print(term)
        
        if term:
            return self.curr_state, self.get_reward(), 0 , term, {'No Info Implemented yet'}
        else:
            return self.curr_state, reward, aux_reward , term, {'No Info Implemented yet'}
        
    def get_reward(self):
        if self.position >= 0.5:
            return 1
        else:
            return 0

    def get_aux_reward(self,action = None,  use_aux = False):
        '''
        Give high velocity positive reward
        '''
        # print)(ac)
        if use_aux:
            return 0
        else:
            return 0
        
    def is_terminal(self):
        if self.position >= 0.5:
            return 1
        elif self.steps_taken >= self.max_horizon :
            return 1
        else:
            return 0
    
    def generate_all_states(self):
        # return stupid state
        return np.array([[0.6, 0.0]])


if __name__ == '__main__':
    rewards_list = []
    env = MountainCarSparse()
    for i in range(50):
        rewards = 0
        done = False
        env.reset()
        while not done:
            action = np.random.randint(env.n_actions)
            # print(action)
            next_state, r, aux_r, done, _ = env.step(action)
            print(r, aux_r)
            rewards += r
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))
