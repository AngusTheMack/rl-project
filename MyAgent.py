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
        # We have discretised the action space, thus we need to makesure that we do the same for our inputs
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        # We have use PyTorchFrrame, so we need to do the same for our observation space
        shape = observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, shape[0], shape[1]), dtype=np.uint8)
        replay_buffer = ReplayBuffer(int(5e3))
        self.agent = DQNAgent(
            self.observation_space,
            self.action_space,
            replay_buffer,
            use_double_dqn=True,
            lr=1e-4,
            batch_size=32,
            gamma=0.99,
        )
        # Set actions
        self.actions = HUMAN_ACTIONS
        # Load Weights
        self.agent.policy_network.load_state_dict(torch.load("model.pth",map_location=torch.device(device)))

    def act(self, observation):
        """
        When we get an obs it will be in RGB, we need to convert it to grayscale because we trained in grayscale

        Then, we need to reshape the new image to be of size (1, 84, 84) instead of (84, 84 before passing it to our model)
        """
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        shape =  observation.shape
        observation = observation.reshape(1, shape[0], shape[1])
        action = self.agent.act(observation)
        return self.actions[action]
