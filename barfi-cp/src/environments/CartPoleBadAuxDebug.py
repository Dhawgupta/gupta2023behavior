'''
Dense reward implementation of MC, where the reward is -1 at every time step
'''
import sys , os, time
sys.path.append(os.getcwd())
import numpy as np
from src.utils.utils import Space
import gym 

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class CartPoleBadAuxDebug(object):
    def __init__(self, action_type = 'discrete', n_actions = 2):
        
        '''
        The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
        | Num | Observation           | Min                  | Max                |
        |-----|-----------------------|----------------------|--------------------|
        | 0   | Cart Position         | -4.8                 | 4.8                |
        | 1   | Cart Velocity         | -Inf                 | Inf                |
        | 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
        | 3   | Pole Angular Velocity | -Inf                 | Inf                |

         The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
        -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

        self.maxIn = np.array([3, 3.5, 0.25, 3.5])
        # self.minIn = env.observation_space.low
        self.minIn = np.array([-3, -3.5, -0.25, -3.5])

        Final max and min range
        Max = (2.4, 3.5, .2095, 3.5)
        Min = (-2.4, -3.5, -.2095, -3.5)
        '''
        self.observation_space = Space(low = np.array([-2.4, -3.5, -.2095, -3.5]), high=  np.array([2.4, 3.5, .2095, 3.5]), dtype = np.float32)
        self.episode = 0
        self.action_space = Space(size = n_actions)
        self.max_horizon = 500
        # self.aux_vel_scale = 10
        self.n_actions = n_actions
        self.env = gym.make('CartPole-v1')
        self.desired_state = np.array([0, 0, 0, 0])
        self.desired_mask = np.array([0, 0, 1, 0])
        # self.P, self.I, self.D = 0.1, 0.01, 0.5
        # self.P, self.D = 0.1, 0.05 # performance around 72
        self.P, self.D = 0.1, 0.5  # performance around 500
        self.reset()

    def seed(self, seed):  
        self.env.seed(seed)
        self.seed = seed

    # Done till here. 
    
    def reset(self):
        self.episode += 1
        self.steps_taken  = 0
        state = self.env.reset()
        self.curr_state = state
        # self.curr_state = np.ar?ray([self.position, self.velocity])
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0
        return self.curr_state.copy()
    
    def step(self, action):
        self.steps_taken += 1
        s_next, reward, term, info = self.env.step(action)
       
        if term :
            return s_next, reward, 0,  term , info

        aux_reward = self.get_aux_reward(action=action)
        self.curr_state = s_next
        return self.curr_state, reward, aux_reward, term, info

    def get_aux_reward(self, action = None, use_aux = True):
        '''
        Give high velocity positive reward
        '''
        if use_aux : 
            pid_action = self.PID_controller(self.curr_state)
            # misleading auxiliary reward
            if action == pid_action:
                return -5
            else:
                return 1

            # return np.abs(self.velocity) * self.aux_vel_scale
        else:
            return 0
        
    # def is_terminal(self):
    #     if self.position >= 0.5:
    #         return 1
    #     elif self.steps_taken >= self.max_horizon :
    #         return 1
    #     else:
    #         return 0
    
    def generate_all_states(self):
        # return stupid state
        return np.array([[0.0, 0.0, 0.0, 0.0]])

    def PID_controller(self, state):
        '''
        Only uses the PD controller
        '''
        error = state - self.desired_state
        self.integral += error
        self.derivative = error - self.prev_error
        self.prev_error = error
        pid = np.dot(self.P * error  + self.D * self.derivative, self.desired_mask)

        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)
        return action



if __name__ == '__main__':
    rewards_list = []
    env = CartPoleBadAuxDebug()
    for i in range(50):
        rewards = 0
        done = False
        env.reset()
        while not done:
            action = np.random.randint(env.n_actions)
            # print(action)
            next_state, r, aux_r, done, _ = env.step(action)
            # print(r, aux_r)
            rewards += r
        rewards_list.append(rewards)
    print(rewards_list)
    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))
