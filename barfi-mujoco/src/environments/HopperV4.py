'''
Dense reward implementation of MC, where the reward is -1 at every time step
'''
import sys , os, time
sys.path.append(os.getcwd())
import numpy as np
from src.utils.utils import Space
import gym 
import copy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class HopperV4(object):
    def __init__(self, action_type = 'continuous', n_actions = 3):

        self.env = gym.make('Hopper-v4')
        # self.observation_space = Space(low = np.array([-2.4, -3.5, -.2095, -3.5]), high=  np.array([2.4, 3.5, .2095, 3.5]), dtype = np.float32)
        # self.action_space = Space(size=n_actions)

        # Set the Mujoco Env Specific variables
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.episode = 0

        self.max_horizon = self.env._max_episode_steps
        # self.aux_vel_scale = 10
        self._seed = None
        self.n_actions = n_actions
        self.reset()

    def seed(self, seed):  
        # self.env.seed(seed)
        self._seed = seed

    # Done till here. 
    
    def reset(self):
        self.episode += 1
        self.steps_taken  = 0
        state = self.env.reset(seed = self._seed)
        self.curr_state = state
        # self.curr_state = np.ar?ray([self.position, self.velocity])
        return copy.copy(self.curr_state)[0]
    
    def step(self, action):
        self.steps_taken += 1
        s_next, reward, term, _,info = self.env.step(action)
        if self.steps_taken >= self.max_horizon:
            term = True
        if term :
            return s_next, reward, 0,  term , info

        aux_reward = self.get_aux_reward(action=action)
        self.curr_state = s_next
        return self.curr_state, reward, aux_reward, term, info

    def get_aux_reward(self, action = None, use_aux = False):
        '''
        Give high velocity positive reward
        '''
        return 0
            # return 0
        
    # def is_terminal(self):
    #     if self.position >= 0.5:
    #         return 1
    #     elif self.steps_taken >= self.max_horizon :
    #         return 1
    #     else:
    #         return 0
    
    def generate_all_states(self):
        # return stupid state
        return np.zeros(self.observation_space.shape).reshape([1,-1])

        # return np.array([[0.0, 0.0, 0.0, 0.0]])

if __name__ == '__main__':
    rewards_list = []
    env = HopperV4()
    for i in range(50):
        rewards = 0
        done = False
        env.reset()
        while not done:
            action = np.random.randint(env.n_actions)
            # print(action)
            next_state, r, aux_r, done, info = env.step(action)
            # print(r, aux_r)
            rewards += r
        rewards_list.append(rewards)
    print(rewards_list)
    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))
