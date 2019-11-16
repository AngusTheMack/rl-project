from AbstractAgent import AbstractAgent

from dqn.agent import DQNAgent
from dqn.wrappers import *
from dqn.replay_buffer import ReplayBuffer
import torch
import gym
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUMAN_ACTIONS = (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
NUM_ACTIONS = len(HUMAN_ACTIONS)
class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = observation_space
        replay_buffer = ReplayBuffer(int(5e3))
        self.agent = DQNAgent(
            observation_space,
            self.action_space,
            replay_buffer,
            use_double_dqn=True,
            lr=1e-4,
            batch_size=32,
            gamma=0.99,
        )
        self.actions = HUMAN_ACTIONS
        self.agent.policy_network.load_state_dict(torch.load("model.pth",map_location=torch.device(device)))

    def act(self, observation):
        action = self.agent.act(observation)
        # print(action, self.actions[action])
        return self.actions[action]
