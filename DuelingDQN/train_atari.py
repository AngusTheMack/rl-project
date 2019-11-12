import torch
import argparse
import os
import numpy as np
import random
from duelingdqn.agent import DuelingAgent
from duelingdqn.wrappers import NoopResetEnv, wrap_deepmind, wrap_pytorch, MaxAndSkipEnv, TimeLimit
import gym
parser = argparse.ArgumentParser(description='Dueling DQN Atari')
parser.add_argument('--load-checkpoint-file', type=str, default=None,
                    help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
parser.add_argument("--env", choices=['boxing', 'pinball', 'breakout'],
                    help="The environment the experiment will be run with, default is boxing", default='boxing')
parser.add_argument("--save_freq", type=int, help="Save checkpoint and videos after this many episodes", default='100')
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

    if (args.load_checkpoint_file):
        eps_start = 0.01
    else:
        eps_start = 1

    env_name = "BoxingNoFrameskip-v4"  # Set Default
    if args.env == 'breakout':
        env_name = 'BreakoutNoFrameskip-v4'
    elif args.env == 'pinball':
        env_name = 'VideoPinballNoFrameskip-v0'

    if not os.path.exists("results/"):
        os.makedirs("results")

    save_loc = "results/" + args.env + "_" + str(1)
    if os.path.exists(save_loc):
        counter = 1
        while True:
            counter += 1
            new_save_loc = "results/" + args.env + "_" + str(counter)
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
        "max_episode_steps": None,
    }
    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    env = gym.make(hyper_params["env"])
    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if hyper_params['max_episode_steps'] is not None:
        env = TimeLimit(env, max_episode_steps=hyper_params['max_episode_steps'])
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    env = gym.wrappers.Monitor(
        env=env, directory=os.path.join(save_loc, "videos"), video_callable=lambda episode_id: episode_id % args.save_freq == 0, force=True
    )
    

    MAX_FRAMES = 1000000
    BATCH_SIZE = 32
    agent = DuelingAgent(env, use_conv=True)
    if torch.cuda.is_available():
        print("Using GPU")
        agent.model.cuda()

    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(torch.load(args.load_checkpoint_file))


    episode_rewards = [0.0]
    episode_reward = 0

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward+=reward
        episode_rewards[-1] += reward
        if len(agent.replay_buffer) > BATCH_SIZE:
            agent.update(BATCH_SIZE)

        
        if done:
            state = env.reset()
            episode_rewards.append(0.0)   
        num_episodes = len(episode_rewards)
        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params["print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            # print("exploitation rate: {}".format(fraction))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            # print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
        if done and len(episode_rewards) % args.save_freq == 0:
            torch.save(agent.policy_network.state_dict(), os.path.join(save_loc,"checkpoint.pth"))
            np.savetxt(os.path.join(save_loc, "rewards.csv"), episode_rewards, delimiter=",")
        state = next_state
    torch.save(agent.policy_network.state_dict(), os.path.join(save_loc,"checkpoint.pth"))
    np.savetxt(os.path.join(save_loc, "rewards.csv"), episode_rewards, delimiter=",")