'''
Dense reward implementation of MC, where the reward is -1 at every time step
'''
import sys, os, time

sys.path.append(os.getcwd())
import numpy as np
from src.utils.utils import Space
import copy
import gym


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class HalfCheetahV4(object):
    def __init__(self, action_type='continuous', n_actions=6):
        # set both weights to 1 so that we can change them later.
        # self.env = gym.make('HalfCheetah-v4', ctrl_cost_weight = 1.0, forward_reward_weight=1.0 )
        self.env = gym.make('HalfCheetah-v4', ctrl_cost_weight = 1.0, forward_reward_weight=1.0 )
        # self.observation_space = Space(low = np.array([-2.4, -3.5, -.2095, -3.5]), high=  np.array([2.4, 3.5, .2095, 3.5]), dtype = np.float32)
        # self.action_space = Space(size=n_actions)

        # Set the Mujoco Env Specific variables
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.episode = 0

        self.max_horizon = self.env._max_episode_steps
        # self.aux_vel_scale = 10
        self.n_actions = n_actions
        self._seed = None
        self.reset()

    def seed(self, seed):
        # self.env.seed(seed)
        self._seed = seed

    # Done till here.

    def reset(self):
        self.episode += 1
        self.steps_taken = 0
        state = self.env.reset(seed = self._seed)
        self.curr_state = state
        # self.curr_state = np.ar?ray([self.position, self.velocity])
        return copy.copy(self.curr_state)[0]

    def step(self, action):
        self.steps_taken += 1
        s_next, reward, term, _ , info = self.env.step(action)
        # print(term)
        if self.steps_taken >= self.max_horizon:
            term = True
        reward_run = info['reward_run']
        reward_ctrl = info['reward_ctrl']

        if term:
            return s_next, reward_run, reward_ctrl, term, info

        # aux_reward = self.get_aux_reward(action=action)
        aux_reward = reward_ctrl

        self.curr_state = s_next
        return self.curr_state, reward_run, aux_reward, term, info

    def get_aux_reward(self, action=None, use_aux=False):
        '''
        Give high velocity positive reward
        '''
        return np.sum(np.square(action))
        # return 0



    def generate_all_states(self):
        # return stupid state
        return np.zeros(self.observation_space.shape).reshape([1, -1])

        # return np.array([[0.0, 0.0, 0.0, 0.0]])


if __name__ == '__main__':
    rewards_list = []
    env = HalfCheetahV4()
    for i in range(50):
        rewards = 0
        done = False
        env.reset()
        iter=0
        while not done:
            action = env.action_space.sample()
            # print(action)
            next_state, r, aux_r, done, info = env.step(action)
            # print(r, aux_r)
            # print(r, aux_r)
            rewards += r
            iter+=1
            print(iter)
        print("Episode Done")
        rewards_list.append(rewards)
    print(rewards_list)
    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))
