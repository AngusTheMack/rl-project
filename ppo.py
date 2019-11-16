import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
import random
import argparse
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HUMAN_ACTIONS = (18, 6, 12, 36, 24, 30)
NUM_ACTIONS = len(HUMAN_ACTIONS)


class HumanActionEnv(gym.ActionWrapper):
    """
    An environment wrapper that limits the action space to
    looking left/right, jumping, and moving forward.
    """

    def __init__(self, env):
        super().__init__(env)
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()


        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(device).flatten()

        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return  np.argmax(action)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def main():

    parser = argparse.ArgumentParser(description='PPO Atari')
    parser.add_argument('--checkpoint', type=str, default=None, help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--lr',type=float,default=1e-4,help="learning rate")
    # parser.add_argument('--continue', action='store_true')
    args = parser.parse_args()

    i = 0
    if not os.path.exists("results"):
        os.mkdir("results")
    while True:
        file_name = "results/experiment_"+str(i)
        if not os.path.exists(file_name):
            dir_to_make = file_name
            break
        i+=1

    os.mkdir(dir_to_make)
    save_loc = dir_to_make+"/"
    print("Saving results to", dir_to_make)
    ############## Hyperparameters ##############
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 50      # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 512         # max timesteps in one episode
    n_latent_var = 2           # number of variables in hidden layer
    update_timestep = 1024      # update policy every n timesteps
    lr = 0.0004
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 8                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = args.seed
    #############################################

    np.random.seed(random_seed)
    random.seed(random_seed)
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    worker_id = int(np.random.randint(999, size=1))
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', docker_training=False, worker_id=worker_id,retro=True, realtime_mode=False, config=config, greyscale=True)
    env.seed(args.seed)
    env = PyTorchFrame(env)
    env = FrameStack(env, 10)
    env = HumanActionEnv(env)

    memory = Memory()
    env_shape = env.observation_space.shape
    state_dim = np.prod(env_shape)
    action_dim = env.action_space.n
    n_latent_var = 600
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    if(args.checkpoint):
        print(f"Loading a policy - { args.checkpoint } ")
        ppo.policy.load_state_dict(torch.load(args.checkpoint))

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(np.array(state), memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            torch.save(ppo.policy.state_dict(), os.path.join(save_loc, "checkpoint_"+str(i_episode)+"_eps.pth"))
            print("Saved models after",i_episode)
    torch.save(ppo.policy.state_dict(), os.path.join(save_loc, "final_checkpoint.pth"))


if __name__ == '__main__':
    main()
