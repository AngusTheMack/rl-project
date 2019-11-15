import random
import numpy as np
import gym
import torch
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import argparse
from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None,
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')

    args = parser.parse_args()
    hyper_params = {
        "seed": 42,  # which seed to use
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
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    worker_id = int(np.random.randint(999, size=1))
    print(worker_id)
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', docker_training=False, worker_id=worker_id, retro=True, realtime_mode=False, config=config)
    # assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env.seed(hyper_params["seed"])
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 2)

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
    print(env.action_space)
    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(torch.load(args.load_checkpoint_file))
    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    step_count = 0
    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()
        if sample > eps_threshold:
            action = agent.act(np.array(state))
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        step_count +=1
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
        if t % 10000:
            torch.save(agent.policy_network.state_dict(), "./checkpoints/checkpoint.pth")

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params["print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), "./checkpoints/checkpoint.pth")
            np.savetxt("rewards.csv", episode_rewards, delimiter=",")
