from AbstractAgent import AbstractAgent

import cv2
import numpy as np
import torch
import random
import gym
from ppo import PPO
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dqn.agent import DQNAgent
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        # self.k = 3
        # self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        shape = observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)
        self.action_space = action_space
        self.memory = ReplayBuffer(int(5e3))
        self.policy_network = DQN(self.observation_space, self.action_space)

        # self.actions = HUMAN_ACTIONS
        # # Load Weights
        # self.framestack = None
        self.policy_network.load_state_dict(torch.load("trf",map_location=torch.device(device)))
        self.policy_network.eval()

    def act(self, observation):
        """
        Then, we need to reshape the new image to be of size (1, 84, 84) instead of (84, 84 before passing it to our model)

        Don't trust this, now we do some tings with stacking frames yo
        """
        # observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # shape =  observation.shape
        # if self.framestack is None:
        #     self.framestack = np.array([observation] * self.k)
        # else:
        #     self.framestack = np.append(self.framestack[1:], observation)
        #     self.framestack = self.framestack.reshape(self.k, 84, 84)
        # shape =  observation.shape
        # observation = observation.reshape(shape[-1], shape[0], shape[1])
        # sample = random.random()
        # if sample > 0.01:
        #     action = self.agent.act(observation)
        # else:
        #     action = self.action_space.sample()
        state = observation
        state = np.rollaxis(state, 2)
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            chosen_action = action.item()
            # print(chosen_action)

            return chosen_action
