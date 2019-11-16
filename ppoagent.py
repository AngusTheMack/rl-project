from AbstractAgent import AbstractAgent


from dqn.agent import DQNAgent
from dqn.wrappers import *
from dqn.replay_buffer import ReplayBuffer
import torch
import gym
from ppo import PPO
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUMAN_ACTIONS = (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)
NUM_ACTIONS = len(HUMAN_ACTIONS)


class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        shape = observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, shape[0], shape[1]), dtype=np.uint8)
        env_shape = self.observation_space.shape

        state_dim = np.prod(env_shape)
        self.state_dim = state_dim
        print("State Dim: ", state_dim)

        self.action_dim =  self.action_space.n
        self.agent = PPO(
            self.state_dim,
            self.action_dim,
            n_latent_var=1000,
            betas = (0.9, 0.999),
            lr=1e-4,
            K_epochs = 4,
            gamma=0.99,
            eps_clip=0.2,
        )
        self.actions = HUMAN_ACTIONS
        self.agent.policy.load_state_dict(torch.load("model.pth",map_location=torch.device(device)))

    def act(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        shape =  observation.shape
        observation = observation.reshape(1, shape[0], shape[1])
        action = self.agent.policy.act(observation)
        return self.actions[action]
