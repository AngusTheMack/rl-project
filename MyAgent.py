from AbstractAgent import AbstractAgent

import cv2
import numpy as np
import torch
import gym
from ppo import PPO
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUMAN_ACTIONS = (18, 6, 12, 36, 24, 30)
NUM_ACTIONS = len(HUMAN_ACTIONS)


class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.k = 10
<<<<<<< HEAD
        self.actions = HUMAN_ACTIONS
=======
        # We have discretised the action space, thus we need to makesure that we do the same for our inputs
>>>>>>> 2eb08ce1cb70f2b7788620253bcc2f23a33fe3e3
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        shape = observation_space.shape
<<<<<<< HEAD
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, shape[0], shape[1]), dtype=np.uint8)
        env_shape = self.observation_space.shape
        state_dim = np.prod(env_shape)
        self.state_dim = state_dim

        self.action_dim =  self.action_space.n
        self.agent = PPO(
            self.state_dim*self.k,
            self.action_dim,
            n_latent_var=600,
            betas = (0.9, 0.999),
=======
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.k, shape[0], shape[1]), dtype=np.uint8)
        print(self.observation_space)
        replay_buffer = ReplayBuffer(int(5e3))
        self.agent = DQNAgent(
            self.observation_space,
            self.action_space,
            replay_buffer,
            use_double_dqn=True,
>>>>>>> 2eb08ce1cb70f2b7788620253bcc2f23a33fe3e3
            lr=1e-4,
            K_epochs = 8,
            gamma=0.99,
            eps_clip=0.2,
        )
        self.actions = HUMAN_ACTIONS
<<<<<<< HEAD
        self.agent.policy.load_state_dict(torch.load("model.pth",map_location=torch.device(device)))
        self.framestack = None
=======
        # Load Weights
        self.framestack = None
        self.agent.policy_network.load_state_dict(torch.load("results/experiment_23/checkpoint_280_eps.pth",map_location=torch.device(device)))
>>>>>>> 2eb08ce1cb70f2b7788620253bcc2f23a33fe3e3


<<<<<<< HEAD
    def act(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.framestack is None:
            self.framestack = np.array([observation] * self.k)
        else:
            self.framestack = self.framestack.reshape(self.k,7056)
            self.framestack = np.append(self.framestack[1:], observation)

        action = self.agent.policy.act(self.framestack.reshape(self.k,7056))
=======
        Then, we need to reshape the new image to be of size (1, 84, 84) instead of (84, 84 before passing it to our model)

        Don't trust this, now we do some tings with stacking frames yo
        """
        print("input obs:",observation.shape)
        print(self.observation_space)
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        shape =  observation.shape
        if self.framestack is None:
            self.framestack = np.array([observation] * self.k)
        else:
            self.framestack = np.append(self.framestack[1:], observation)
            self.framestack = self.framestack.reshape(self.k, 84, 84)
        action = self.agent.act(self.framestack)
>>>>>>> 2eb08ce1cb70f2b7788620253bcc2f23a33fe3e3
        return self.actions[action]
