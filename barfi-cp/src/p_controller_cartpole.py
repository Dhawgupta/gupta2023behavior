from http.client import EXPECTATION_FAILED
import numpy as np
import gym
import sys

def float_to_string(number, precision=10):
    return '{0:.{prec}f}'.format(
        number, prec=precision,
    ).rstrip('0').rstrip('.') or '0'

if len(sys.argv) < 2:
    raise Exception()

P = float(sys.argv[1])
# I = float(sys.argv[2])
# D = float(sys.argv[3])
NUM_EPISODES = 100

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
np.random.seed(0)
env = gym.make('CartPole-v1')
desired_state = np.array([0, 0, 0, 0])
desired_mask = np.array([0, 0, 1, 0])

# P, I, D = 0.1, 0.01, 0.5
returns = np.zeros(NUM_EPISODES)


for i_episode in range(NUM_EPISODES):
    ret_ = 0
    state = env.reset()
    # integral = 0
    # derivative = 0
    # prev_error = 0
    for t in range(500):
        # env.render()
        error = state - desired_state

        # integral += error
        # derivative = error - prev_error
        # prev_error = error

        pid = np.dot(P * error , desired_mask)
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)

        state, reward, done, info = env.step(action)
        ret_ += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            returns[i_episode] = ret_
            break
    returns[i_episode] = ret_

env.close()

file_name = f'P_results/{float_to_string(P)}.npy'
np.save(file_name, returns)