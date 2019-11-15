from AbstractAgent import AbstractAgent

from dqn.agent import DQNAgent
from dqn.wrappers import *
class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        # TODO Initialise your agent's models

        # for example, if your agent had a Pytorch model it must be load here
        # model.load_state_dict(torch.load( 'path_to_network_model_file', map_location=torch.device(device)))
        self.agent = DQNAgent(
            observation_space,
            action_space,
            replay_buffer,
            use_double_dqn=True,
            lr=1e-4,
            batch_size=32,
            gamma=0.99,
        )
        self.agent.policy_network.load_state_dict(torch.load(args.load_checkpoint_file))
    def act(self, observation):
        # Perform processing to observation
        self.agent.act(observation)
        # # TODO: return selected action
        # return self.action_space.sample()
