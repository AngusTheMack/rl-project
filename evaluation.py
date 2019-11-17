from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import numpy as np
import sys


def run_episode(env):

    from MyAgent import MyAgent
    agent = MyAgent(env.observation_space, env.action_space)

    done = False
    episode_return = 0.0
    state = env.reset()
    while not done:
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        episode_return += reward
        state = new_state
    return episode_return


if __name__ == '__main__':
    error_occurred = False

    eval_seeds = [1, 2, 3, 4, 5]

    # Create the ObstacleTowerEnv gym and launch ObstacleTower
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    worker_id = int(np.random.randint(999, size=1))
    print(worker_id)
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', docker_training=False, worker_id=worker_id, retro=True,
                           realtime_mode=False, config=config, greyscale=False)
    env = ObstacleTowerEvaluation(env, eval_seeds)

    while not env.evaluation_complete:
        # Deleted the try catch because the error txt file was confusing
        episode_rew = run_episode(env)

    env.close()
    if error_occurred:
        print(-100.0)
    else:
        print(env.results['average_reward']*10000)
