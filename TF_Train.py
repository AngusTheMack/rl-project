# Vanilla imports
import argparse
import os
import random
import copy
import numpy as np
import tensorflow as tf

# Custom Imports
from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
from wrappers import *
from ModelPPO import PPO


### Global Variables
# Select actions as described by @see
HUMAN_ACTIONS = (18, 6, 12, 36, 24, 30)
NUM_ACTIONS = len(HUMAN_ACTIONS)


# We are defining the function to get the Generalized Advantage Esstimation
def get_gaes(rewards, state_values, next_state_values, GAMMA, LAMBDA):
    deltas = [r_t + GAMMA * next_v - v for r_t, next_v, v in zip(rewards, next_state_values, state_values)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
    return gaes, deltas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO')
    parser.add_argument('--checkpoint', type=str, default=None, help='Where checkpoint file if its a directory, then loads the most recent - otherwise loads the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--lr',type=float,default=(5 * 10e-5),help="learning rate")
    parser.add_argument('--save_freq',type=int, default=50, help="Save model after n episodes")
    args = parser.parse_args()

    # Make a new directory to not overwrite prev results
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
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }

    worker_id = int(np.random.randint(999, size=1))
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', docker_training=False, worker_id=worker_id,retro=True, realtime_mode=False, config=config, greyscale=True)
    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = PyTorchFrame(env) # Change Name
    # env = FrameStack(env, 10)
    env = HumanActionEnv(env)


    state = env.reset()


    # Defines shapes for placeholders in tf graphs
    state_shape = state.shape
    frame_height = state.shape[1]
    frame_width = state.shape[2]
    channels = state.shape[0]
    n_actions = NUM_ACTIONS

    print("Overall state space:",state_shape)
    print("Height:",frame_height, "Width",frame_width, "Channels",channels)

    # Number of actions to adjust policy over
    timesteps = 4

    # Some hyperparameters for the optimization
    epsilon = 0.2
    learning_rate = args.lr
    GAMMA = 0.99
    LAMBDA = 0.95

    # Initialize the model in TF

    tf.reset_default_graph()
    sess = tf.Session()
    model = PPO(sess, frame_height, frame_width, channels, n_actions, timesteps, epsilon, learning_rate, GAMMA, LAMBDA)


    sess.run(tf.global_variables_initializer())
    checkpoint_saver = tf.compat.v1.train.Saver()
    if args.checkpoint is not None:
        if os.path.isdir( args.checkpoint):
            print("Directory given, loading most recent checkpoint from", args.checkpoint)
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        else:
            checkpoint = args.checkpoint
        print("Restoring from",checkpoint)
        checkpoint_saver.restore(sess, checkpoint)
        print("Restored successfully")
    total_episodes = 10000000
    episode_counter = 0
    done = False

    # Normalise state , change to 10 for frame stacked
    current_state = state.reshape(1, frame_height, frame_width, channels) / 255
    # print("Current_state", current_state.shape)
    # Track episode values
    states = []
    prev_actions_list = []
    actions = []
    state_values = []
    rewards = []

    # Number of epochs for PPO and batch size
    epochs = 2
    batch_size = 64

    # Some performance stats to keep track of
    performance = []
    max_reward = -1000
    max_reward_tracker = []
    max_score = -1000
    max_score_tracker = []

    while episode_counter < total_episodes:
        if done:
            # AVG Score
            score = np.mean(rewards)*10000
            if score > max_score:
                max_score = score
                max_score_tracker.append(max_score)
                name = os.path.join(save_loc, "best")
                checkpoint_saver.save(sess, name, global_step=0)
                print("Saved new best episode after",episode_counter,"episodes")
            next_state_values = state_values[1:] + [0]
            GAEs, deltas = get_gaes(rewards, state_values, next_state_values, GAMMA, LAMBDA)

            # Change to np arrays
            # print(state.shape)

            states = np.reshape(states, newshape = [-1] + list(state.shape))
            # print("Before being a badass", states.shape)
            states = states.reshape(states.shape[0],states.shape[2], states.shape[3], states.shape[1])
            # print("Before update",states.shape)
            prev_actions_list = np.array(prev_actions_list)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_state_values = np.array(next_state_values)
            GAEs = np.array(GAEs)

            # Update Prev Model
            model.update()

            # Train the Model
            model.train(states, actions, GAEs, rewards, next_state_values, prev_actions_list, epochs, batch_size)

            # Append most recent score to the performance tracker and reinitialize everything for the next episode
            performance.append(score)
            state = env.reset()
            current_state = state.reshape(1, frame_height, frame_width, channels) / 255
            # print("Current state after training", current_state.shape)
            states = []
            prev_actions_list = []
            actions = []
            state_values = []
            rewards = []

            # Print out a few numbers to keep track of things
            print("Episode: {}, Score: {}, Max Score {}, Max Reward: {}".format(episode_counter, performance[-1], max_score_tracker[-1], max_reward_tracker[-1]))
            if episode_counter % args.save_freq == 0:
                name = os.path.join(save_loc, "model")
                checkpoint_saver.save(sess, name, global_step=episode_counter)
                print("Saved Checkpoint after",episode_counter,"episodes")
            # Maybe add another save ting for increasing accuracy
            episode_counter += 1

        # Get your inputs ready for the model
        prev_state = np.copy(state)
        prev_actions = np.array([0]*(timesteps - min(len(actions), timesteps)) + actions[-timesteps:])
        prev_actions = np.hstack((np.eye(n_actions)[prev_actions[0]], np.eye(n_actions)[prev_actions[1]],
                                  np.eye(n_actions)[prev_actions[2]], np.eye(n_actions)[prev_actions[3]]))

        # Extract the policy and state-value from the model
        policy, value = sess.run([model.policy, model.value], {model.inputs: current_state,
                                 model.previous_actions: np.expand_dims(prev_actions, axis = 0), model.training: False})

        # Policy is a probabilistic output, so we sample from it to get our action
        action = np.random.choice(np.arange(n_actions), 1, p = policy.reshape(-1))[0]

        # We take the action in the environment and receive the next state, reward, and some other useful info
        state, reward, done, info = env.step(action)

        # Reformat our state to use as input to get the next action
        current_state = state.reshape(1, frame_height, frame_width, channels) / 255
        # print("Current state before next action:", current_state.shape)
        # The max absolute reward we can get is 15, so we make sure the reward is between -1 and 1
        # reward /= 15


        # Appending the data to be used in training
        states.append(prev_state)
        prev_actions_list.append(prev_actions)
        actions.append(action)
        state_values.append(value[0,0])
        rewards.append(reward)

        # Tracking the max position in the level
        if reward > max_reward:
            max_reward = reward
        max_reward_tracker.append(max_reward)

        # This is to render the actual game (Comment this out for a speed boost in training)
        # env.render()

    env.close()
