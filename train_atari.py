import random
import numpy as np
import gym
import os
import torch
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from wrappers import *
import argparse
from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN for Obstacle Tower')
    parser.add_argument('--checkpoint', type=str, default=None, help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for training')
    parser.add_argument('--lr',type=float,default=1e-4,help="learning rate")
    args = parser.parse_args()


    # Create a unique folder to store results and checkpoints
    i = 0
    if not os.path.exists("results"):
        os.mkdir("results")
    while True:
        file_name = "results/experiment_"+str(i)
        if not os.path.exists(file_name):
            dir_to_make = file_name
            break
        i+=1

    os.mkdir(dir_to_make)
    save_loc = dir_to_make+"/"
    print("Saving results to", dir_to_make)


    # Hyperparams for Experiment
    hyper_params = {
        "discount-factor": 0.9,  # discount factor
        "num-steps":  5000000,  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }

    # Set a random seed randomly, or using the inputted one
    if args.seed is None:
        random_seed = int(np.random.randint(999, size=1))
    else:
        random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)


    # Create the environment
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    worker_id = int(np.random.randint(999, size=1))
    print(worker_id)
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', docker_training=False, worker_id=worker_id, retro=True,
                            realtime_mode=False, config=config)
    env.seed(random_seed)

    # Run with specific wrappers #
    # This is the only Wrapper we used, as the others were didn't add enough value
    env = PyTorchFrame(env)
    # env = FrameStack(env, 3)
    # env = HumanActionEnv(env)

    # Create Agent to Train
    replay_buffer = ReplayBuffer(int(5e3))
    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=True,
        lr=args.lr,
        batch_size=hyper_params["batch-size"],
        gamma=hyper_params["discount-factor"],
    )

    # If we have prereained weights, load them
    if(args.checkpoint):
        print(f"Loading a policy - { args.checkpoint } ")
        agent.policy_network.load_state_dict(torch.load(args.checkpoint))

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    step_count = 0
    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        eps_threshold = 0.01
        fraction = min(1.0, float(t) / eps_timesteps)
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
        if t % 100000 == 0 and t != 0:
            torch.save(agent.policy_network.state_dict(), os.path.join(save_loc, "checkpoint_"+str(t)+"_step.pth"))
            print("Saved Checkpoint after",t,"steps")

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params["print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), os.path.join(save_loc, "checkpoint_"+str(num_episodes)+"_eps.pth"))
            np.savetxt(os.path.join(save_loc,"rewards.csv"), episode_rewards, delimiter=",")
    torch.save(agent.policy_network.state_dict(), os.path.join(save_loc, "final_checkpoint.pth"))
    np.savetxt(os.path.join(save_loc,"rewards.csv"), episode_rewards, delimiter=",")
