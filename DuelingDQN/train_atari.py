
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.autograd as autograd
import gym
import argparse
import os
from duelingdqn.agent import DuelingAgent
from duelingdqn.buffers import * 
from duelingdqn.wrappers import make_atari, wrap_deepmind, wrap_pytorch

parser = argparse.ArgumentParser(description='Dueling DQN Atari')
parser.add_argument('--load-checkpoint-file', type=str, default=None, help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
parser.add_argument("--env",choices=['boxing','pinball','breakout'] ,help="The environment the experiment will be run with, default is boxing", default='boxing')
parser.add_argument("--save_freq",type=int, help="Save checkpoint and videos after this many episodes", default='100')
args = parser.parse_args()

def mini_batch_train_frames(env, agent, max_frames, batch_size):
    episode_rewards = []
    state = env.reset()
    episode_reward = 0

    for frame in range(max_frames):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)   

        if done:
            episode_rewards.append(episode_reward)
            print("Frame " + str(frame) + ": " + str(episode_reward))
            state = env.reset()
            episode_reward = 0
        
        state = next_state
                
    return episode_rewards

if __name__ == "__main__":

	# if(args.load_checkpoint_file):
	# 	eps_start= 0.01
	# else:
	# 	eps_start= 1

	# env_name = "BoxingNoFrameskip-v4" # Set Default
	# if args.env=='breakout':
	# 	env_name = 'BreakoutNoFrameskip-v4'
	# elif args.env == 'pinball':
	# 	env_name = 'VideoPinballNoFrameskip-v0'

	# if not os.path.exists("results/"):
	# 	os.makedirs("results")

	# save_loc = "results/"+args.env+"_"+str(1)
	# if os.path.exists(save_loc):
	# 	counter = 1
	# 	while True:
	# 		counter +=1
	# 		new_save_loc = "results/"+args.env+"_"+str(counter)
	# 		if os.path.exists(new_save_loc):
	# 			continue
	# 		else:
	# 			save_loc = new_save_loc
	# 			break
	# print("Training Model with:", env_name)
	# print("Making directory to save results in", save_loc)
	# os.makedirs(save_loc)

	# hyper_params = {
	# 	"seed": 42,  # which seed to use
	# 	"env": env_name,  # name of the game
	# 	"replay-buffer-size": int(5e3),  # replay buffer size
	# 	"learning-rate": 1e-4,  # learning rate for Adam optimizer
	# 	"discount-factor": 0.99,  # discount factor
	# 	"num-steps": int(1e6),  # total number of steps to run the environment for
	# 	"batch-size": 32,  # number of transitions to optimize at the same time
	# 	"learning-starts": 10000,  # number of steps before learning starts
	# 	"learning-freq": 1,  # number of iterations between every optimization step
	# 	"use-double-dqn": True,  # use double deep Q-learning
	# 	"target-update-freq": 1000,  # number of iterations between every target network update
	# 	"eps-start": 1.0,  # e-greedy start threshold
	# 	"eps-end": 0.01,  # e-greedy end threshold
	# 	"eps-fraction": 0.3,  # fraction of num-steps
	# 	"print-freq": 10,
	# }
	# np.random.seed(hyper_params["seed"])
	# random.seed(hyper_params["seed"])

	# env = gym.make(hyper_params["env"])
	# env.seed(hyper_params["seed"])
	env_id = "PongNoFrameskip-v4"
	env    = make_atari(env_id)
	env    = wrap_deepmind(env)
	env    = wrap_pytorch(env)

	MAX_FRAMES = 1000000
	BATCH_SIZE = 32

	agent = DuelingAgent(env, use_conv=True)
	if torch.cuda.is_available():
		print("Using GPU")
		agent.model.cuda()

	episode_rewards = mini_batch_train_frames(env, agent, MAX_FRAMES, BATCH_SIZE)