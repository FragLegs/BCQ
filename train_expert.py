import numpy as np
import torch
import gym
import argparse
import os
import wandb

import utils
import DDPG

# Shortened version of code originally found at https://github.com/sfujim/TD3

if __name__ == '__main__':
    config = {
        'expert': 'DDPG',
        'computer': os.environ.get('COMPUTER_NAME', 'unknown'),
        'device': DDPG.device.type,
    }

    wandb.init(project='cs234', config=config, job_type=__file__[:-len('.py')])

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Hopper-v2')           # OpenAI gym environment name
    parser.add_argument('--seed', default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--max_timesteps', default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument('--start_timesteps', default=1e3, type=int)  # How many time steps purely random policy is run for
    parser.add_argument('--expl_noise', default=0.1, type=float)     # Std of Gaussian exploration noise
    args = parser.parse_args()

    wandb.config.update(args)

    directory = os.path.join(wandb.run.dir, 'experts')

    # save checkpoints to W&B as we go
    wandb.save(os.path.join(directory, '*chkpt*'))

    file_name = 'DDPG_{}_{}'.format(args.env_name, str(args.seed))
    print('---------------------------------------')
    print('Settings: ' + file_name)
    print('---------------------------------------')

    if not os.path.exists(directory):
        os.makedirs(directory)

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy and buffer
    policy = DDPG.DDPG(state_dim, action_dim, max_action)
    wandb.watch((policy.actor, policy.critic))
    replay_buffer = utils.ReplayBuffer()

    total_timesteps = 0
    episode_num = 0
    done = True
    episode_reward = None
    episode_timesteps = None

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print('Total T: {} Episode Num: {} Episode T: {} Reward: {}'.format(
                    total_timesteps, episode_num, episode_timesteps, episode_reward
                ))

                # log episode to W&B
                wandb.log({
                    'reward': episode_reward,
                    'timesteps': episode_timesteps,
                })

                policy.train(replay_buffer, episode_timesteps)

                # Save policy
                if episode_num % 1000 == 0:
                    policy.save(
                        '{}-chkpt-{}'.format(file_name, episode_num),
                        directory=directory
                    )

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                random = np.random.normal(
                    0, args.expl_noise, size=env.action_space.shape[0]
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

    # Save final policy
    policy.save(file_name, directory=directory)
