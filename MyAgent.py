from AbstractAgent import AbstractAgent

import cv2
import numpy as np
import torch
import random
import gym
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dqn.agent import DQNAgent
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        """
        Takes in the boy
        """
        shape = observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)
        self.action_space = action_space
        self.memory = ReplayBuffer(int(5e3))
        self.policy_network = DQN(self.observation_space, self.action_space)
        self.policy_network.load_state_dict(torch.load("model.pth",map_location=torch.device(device)))
        self.policy_network.eval()

    def act(self, observation):
        """
        Given a state, choose an action according to learned policy policy_network
        @see dqn.model
        """
        observation = np.rollaxis(observation, 2)
        observation = np.array(observation) / 255.0
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(observation)
            _, action = q_values.max(1)
            chosen_action = action.item()
            return chosen_action
