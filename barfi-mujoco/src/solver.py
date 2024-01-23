#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function
# from memory_profiler import profile

import numpy as np
import src.utils.utils as utils
from src.config import Config
from time import time
import matplotlib.pyplot as plt
import torch
import resource
import wandb

class Solver:
    def __init__(self, config):
        # Initialize the required variables
        # make the config for state_lr = actor_lr
        config.state_lr = config.actor_lr
        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)
        # wandb.watch(self.model)

    def train(self, max_episodes):
        # Learn the model on the environment
        return_history = []
        aux_return_history = []
        aux_only_return_history = []
        gamma_history = []
        true_rewards = []
        action_prob = []
        x1_history = []
        x3_history = []
        gradients = []
        entropies = []
        return_auc_mean = 0
        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0
        rm_aux = 0
        rm_only_aux = 0
        steps = 0
        t0 = time()
        with torch.no_grad():
            all_states_numpy = self.env.generate_all_states()
            all_states = torch.tensor(all_states_numpy).float()
        for episode in range(start_ep, max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()
            self.model.reset()

            step, total_r, total_only_aux = 0, 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                action, dist = self.model.get_action(state)
                # print(action, dist)
                new_state, reward, aux_reward, done, info = self.env.step(action=action)
                # print(reward, aux_reward)
                self.model.update(state, action, dist, reward, aux_reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                total_only_aux += aux_reward
                step += 1
                # if step >= self.config.max_steps:
                #     break

            # track inter-episode progress
            # returns.append(total_r)
            aux_return , gamma = self.model.get_aligned_return_gamma()
            # self.model.memory.reset()
            steps += step
            rm = 0.0*rm + 1.0*total_r
            rm_aux = 0.0*rm_aux + 1.0*aux_return
            rm_only_aux = 0.0 * rm_only_aux + 1.0 * total_only_aux
            # rm = total_r
            save_freq = 1
            return_auc_mean = ((1/(episode + 1)) * rm) + (episode/(episode+1)) * return_auc_mean
            if episode%save_freq == 0 or episode == self.config.max_episodes-1:
                x1, x3 = self.model.get_aligned_reward_terms(all_states)
                wandb.log( {
                    'return' : rm,
                    'return_auc' : return_auc_mean,
                    'episode' : episode,
                    'aux_return' : rm_aux,
                    'aux_only_return' : rm_only_aux,
                    'entropies' : self.model.entropy,
                    'gamma' : gamma,
                    # 'gradients' : self.model.get_grads(),
                    'x1' : x1,
                    'x3' : x3,
                    'running_return_rp' : self.model.running_return_outer,
                    'running_return_aux' : self.model.running_return_aux,
                    'memory': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    })
                
                x1_history.append(x1)
                x3_history.append(x3)
                rm_history.append(rm)
                entropies.append(self.model.entropy)
                gradients.append(self.model.get_grads())
                return_history.append(total_r)
                aux_return_history.append(aux_return)
                gamma_history.append(gamma)
                aux_only_return_history.append(rm_only_aux)

                print("{} :: Rewards {:.3f} :: Aux Rewards {:.3f} :: Gamma {:.2f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {} :: x1 : {} :: x3 : {}".
                      format(episode, rm, rm_aux, gamma, steps, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads(), x1, x3))

                # self.model.save()
                # utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))
                # utils.save_plots(gamma_history, config=self.config, name='{}_gamma'.format(self.config.seed))
                # utils.save_plots(aux_return_history, config=self.config, name='{}_aux_rewards'.format(self.config.seed))
            
                t0 = time()
                steps = 0
        self.model.save()
        utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))
        utils.save_plots(gamma_history, config=self.config, name='{}_gamma'.format(self.config.seed))
        utils.save_plots(aux_return_history, config=self.config, name='{}_aux_rewards'.format(self.config.seed))
        np.save(self.config.paths['results'] + f'{self.config.seed}_x1', x1_history)
        np.save(self.config.paths['results'] + f'{self.config.seed}_x3', x3_history)
        np.save(self.config.paths['results'] + f'{self.config.seed}_all_states', all_states_numpy)
        # utils.save_plots(x1_history, config=self.config, name='{}_x1'.format(self.config.seed))
        # utils.save_plots(x3_history, config=self.config, name='{}_x3'.format(self.config.seed))
        data = {'x1' : x1_history,
                'x3': x3_history,
                'entropies' : entropies, 
                'gradients' : gradients,
                'aux_only' : aux_only_return_history} 

        # plt.plot(return_history)
        # plt.savefig('plot.png')
        return return_history, gamma_history, aux_return_history, data

    def eval(self, max_episodes):
        self.model.load()
        temp = max_episodes/100

        returns = []
        for episode in range(max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                # action, dist = self.model.get_action(state)
                action, dist = self.model.get_action_POMDP(state)
                new_state, reward, aux_reward, done, info = self.env.step(action=action)
                state = new_state

                # Tracking intra-episode progress
                total_r += self.config.gamma**step * reward

                step += 1
                # if step >= self.config.max_steps:
                #     break

            returns.append(total_r)
            if episode % temp == 0:
                print("Eval Collected {}/{} :: Mean return {}".format(episode, max_episodes, np.mean(returns)))

                np.save(self.config.paths['results'] + 'eval_returns_' + str(self.config.alpha) + '_' + str(self.config.seed) , returns)


    def collect(self, max_episodes):
        self.model.load()
        temp = max_episodes/100

        trajectories = []
        for episode in range(max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            traj = []
            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                # action, rho = self.model.get_action(state, behavior=True)
                action, rho = self.model.get_action_POMDP(state, behavior=True)
                new_state, reward, aux_reward, done, info = self.env.step(action=action)
                state = new_state

                # Track importance ratio of current action, and current reward
                traj.append((rho, reward))

                step += 1
                # if step >= self.config.max_steps:
                #     break

            # Make the length of all trajectories the same.
            # Make rho = 1 and reward = 0, which corresponds to a self loop in the terminal state
            for i in range(step, self.env.max_horizon):
                traj.append((1, 0))

            trajectories.append(traj)

            if episode % temp == 0:
                print("Beta Collected {}/{}".format(episode, max_episodes))

                np.save(self.config.paths['results'] + 'beta_trajectories_' + str(self.config.alpha) + '_' + str(self.config.seed) , trajectories)
