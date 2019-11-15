from AbstractAgent import AbstractAgent

from dqn.agent import DQNAgent
from dqn.wrappers import *
from dqn.replay_buffer import ReplayBuffer
import torch
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        replay_buffer = ReplayBuffer(int(5e3))
        self.agent = DQNAgent(
            observation_space,
            action_space,
            replay_buffer,
            use_double_dqn=True,
            lr=1e-4,
            batch_size=32,
            gamma=0.99,
        )
        self.agent.policy_network.load_state_dict(torch.load("results/experiment_3/checkpoint_40_eps.pth",map_location=torch.device(device)))


    def act(self, observation):
        return self.agent.act(observation)
