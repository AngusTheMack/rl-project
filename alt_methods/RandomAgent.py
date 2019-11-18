from AbstractAgent import AbstractAgent
"""
Default Agent supplied by our course
"""

class RandomAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()
