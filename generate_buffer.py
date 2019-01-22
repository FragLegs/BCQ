import gym
import numpy as np
import torch
import argparse
import os
import wandb

import utils
import DDPG


if __name__ == '__main__':
    config = {
        'expert': 'DDPG',
        'computer': os.environ.get('COMPUTER_NAME', 'unknown'),
        'device': DDPG.device.type,
    }

    wandb.init(config=config, job_type=__file__[:-len('.py')])

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Hopper-v1')         # OpenAI gym environment name
    parser.add_argument('--seed', default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--buffer_size', default=1e5, type=float)  # Max time steps to run environment for
    parser.add_argument('--noise1', default=0.3, type=float)       # Probability of selecting random action
    parser.add_argument('--noise2', default=0.3, type=float)       # Std of Gaussian exploration noise
    args = parser.parse_args()

    wandb.config.update(args)

    buffers_directory = os.path.join(wandb.run.dir, 'buffers')
    experts_directory = os.path.join(wandb.run.dir, 'experts')

    file_name = 'DDPG_{}_{}'.format(args.env_name, str(args.seed))
    buffer_name = 'Robust_{}_{}'.format(args.env_name, str(args.seed))
    print('---------------------------------------')
    print('Settings: ' + file_name)
    print('---------------------------------------')

    if not os.path.exists(buffers_directory):
        os.makedirs(buffers_directory)

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action)
    policy.load(file_name, experts_directory)

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer()

    total_timesteps = 0
    episode_num = 0
    done = True
    episode_reward = None
    episode_timesteps = None

    while total_timesteps < args.buffer_size:
        if done:
            if total_timesteps != 0:
                print('Total T: {} Episode Num: {} Episode T: {} Reward: {}'.format(
                    total_timesteps, episode_num, episode_timesteps, episode_reward
                ))

                # log episode to W&B
                wandb.log({
                    'episode': episode_num,
                    'reward': episode_reward,
                    'timesteps': episode_timesteps,
                    'total_timesteps': total_timesteps
                })

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Add noise to actions
        if np.random.uniform(0, 1) < args.noise1:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.noise2 != 0:
                random = np.random.normal(
                    0, args.noise2, size=env.action_space.shape[0]
                )
                action = (action + random).clip(
                    env.action_space.low, env.action_space.high
                )

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1

    # Save final buffer
    replay_buffer.save(buffer_name, buffers_directory)
