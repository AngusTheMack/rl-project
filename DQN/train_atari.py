import random
import numpy as np
import gym
import torch
import argparse
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import os
parser = argparse.ArgumentParser(description='DQN Atari')
parser.add_argument('--load-checkpoint-file', type=str, default=None, help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
parser.add_argument("--env",choices=['boxing','pinball','breakout'] ,help="The environment the experiment will be run with, default is boxing", default='boxing')
parser.add_argument("--save_freq",type=int, help="Save checkpoint and videos after this many episodes", default='100')
args = parser.parse_args()
if __name__ == "__main__":

	if(args.load_checkpoint_file):
		eps_start= 0.01
	else:
		eps_start= 1

	env_name = "BoxingNoFrameskip-v4" # Set Default
	if args.env=='breakout':
		env_name = 'BreakoutNoFrameskip-v4'
	elif args.env == 'pinball':
		env_name = 'VideoPinballNoFrameskip-v0'

	if not os.path.exists("results/"):
		os.makedirs("results")

	save_loc = "results/"+args.env+"_"+str(1)
	if os.path.exists(save_loc):
		counter = 1
		while True:
			counter +=1
			new_save_loc = "results/"+args.env+"_"+str(counter)
			if os.path.exists(new_save_loc):
				continue
			else:
				save_loc = new_save_loc
				break

	print("Training Model with:", env_name)
	print("Making directory to save results in", save_loc)
	os.makedirs(save_loc)
	hyper_params = {
		"seed": 42,  # which seed to use
		"env": env_name,  # name of the game
		"replay-buffer-size": int(5e3),  # replay buffer size
		"learning-rate": 1e-4,  # learning rate for Adam optimizer
		"discount-factor": 0.99,  # discount factor
		"num-steps": int(1e6),  # total number of steps to run the environment for
		"batch-size": 32,  # number of transitions to optimize at the same time
		"learning-starts": 10000,  # number of steps before learning starts
		"learning-freq": 1,  # number of iterations between every optimization step
		"use-double-dqn": True,  # use double deep Q-learning
		"target-update-freq": 1000,  # number of iterations between every target network update
		"eps-start": 1.0,  # e-greedy start threshold
		"eps-end": 0.01,  # e-greedy end threshold
		"eps-fraction": 0.3,  # fraction of num-steps
		"print-freq": 10,
	}

	np.random.seed(hyper_params["seed"])
	random.seed(hyper_params["seed"])

	assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
	env = gym.make(hyper_params["env"])
	env.seed(hyper_params["seed"])

	env = NoopResetEnv(env, noop_max=30)
	env = MaxAndSkipEnv(env, skip=4)
	env = EpisodicLifeEnv(env)
	env = FireResetEnv(env)
	env = WarpFrame(env)
	env = PyTorchFrame(env)
	env = ClipRewardEnv(env)
	env = FrameStack(env, 4)
	env = gym.wrappers.Monitor(
		env=env, directory=os.path.join(save_loc, "videos"), video_callable=lambda episode_id: episode_id % args.save_freq == 0, force=True
	)
	replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

	agent = DQNAgent(
		env.observation_space,
		env.action_space,
		replay_buffer,
		use_double_dqn=hyper_params["use-double-dqn"],
		lr=hyper_params["learning-rate"],
		batch_size=hyper_params["batch-size"],
		gamma=hyper_params["discount-factor"],
	)
	if(args.load_checkpoint_file):
		print(f"Loading a policy - { args.load_checkpoint_file } ")
		agent.policy_network.load_state_dict(torch.load(args.load_checkpoint_file))
	eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"]) # T
	episode_rewards = [0.0]

	state = env.reset()
	for t in range(hyper_params["num-steps"]):

		fraction = np.min(np.array([1.0,float(t)/eps_timesteps]))

		eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])
		sample = random.random()
		if sample > eps_threshold:
			action = agent.act(np.array(state))
		else:
			action = env.action_space.sample()
		next_state, reward, done, _ = env.step(action)
		agent.memory.add(state, action, reward, next_state, float(done))
		state = next_state
		episode_rewards[-1] += reward
		if done:
			state = env.reset()
			episode_rewards.append(0.0)

		if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
			agent.optimise_td_loss()

		if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
			agent.update_target_network()

		num_episodes = len(episode_rewards)

		if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params["print-freq"] == 0:
			mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
			print("********************************************************")
			print("steps: {}".format(t))
			print("episodes: {}".format(num_episodes))
			print("exploitation rate: {}".format(fraction))
			print("mean 100 episode reward: {}".format(mean_100ep_reward))
			print("% time spent exploring: {}".format(int(100 * eps_threshold)))
			print("********************************************************")
		if done and len(episode_rewards) % args.save_freq == 0:
			torch.save(agent.policy_network.state_dict(), os.path.join(save_loc,"checkpoint.pth"))
			np.savetxt(os.path.join(save_loc, "rewards.csv"), episode_rewards, delimiter=",")
	torch.save(agent.policy_network.state_dict(), os.path.join(save_loc,"checkpoint.pth"))
	np.savetxt(os.path.join(save_loc, "rewards.csv"), episode_rewards, delimiter=",")
