import gym
import numpy as np
import torch
from policy import PolicyMC, PolicyReinforce, PolicyIntegrationTrapezoidal
from qcritic import QCritic
from vcritic import VCritic
from config import Config
import random
import time
import os
import argparse
from utils import *

# Logging stuff
from tensorboardX import SummaryWriter


def run(env, config, 
        policy_type='integrate',
        seed=7, 
        use_target=False, 
        checkpoint_freq=1000,
        num_episodes=4000):

    env.seed(seed) 
    torch.manual_seed(seed)

    writer = get_writer(policy_type, config, seed)

    vcritic = VCritic(env, config)
    policy = get_policy(policy_type, env, config, writer)

    if policy_type == 'integrate' or policy_type == 'mc':
        use_qcritic = True
        qcritic = QCritic(env, config)
        qcritic.train()
        if use_target:
            target_qcritic = QCritic(env, config)
            target_qcritic.load_state_dict(qcritic.state_dict())
            target_qcritic.eval()
    else:
        use_qcritic = False

    policy.train()
    vcritic.train()

    for episode in range(num_episodes):

        states = []
        actions = []
        rewards = []

        total_steps = 0

        # while total_steps < config.batch_size:

        observation = env.reset()
        done = False
        ep_length = 0

        ep_states = [observation]
        ep_actions = []
        ep_rewards = []
        ep_dones = []

        while not done:
            # env.render()
            action = policy.get_action(observation)
            observation, reward, done, info = env.step(action)
            ep_states.append(observation)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_length += 1
            if(len(ep_actions) >= 2):
                if use_qcritic:
                    if use_target:
                        qcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2], ep_actions[-1], target_q=target_qcritic)
                        q_state_dict = qcritic.state_dict()

                        # target_state_dict = target_qcritic.state_dict()

                        # for name, param in target_state_dict.items():
                        #     if not "weight" in name:
                        #         continue
                        #     param.data = (1-config.tau)*param.data + config.tau*q_state_dict[name].data
                        #     target_state_dict[name].copy_(param)
                        # target_qcritic.load_state_dict(target_state_dict)
                        target_qcritic.load_state_dict(q_state_dict)

                    else:
                        qcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2], ep_actions[-1])

                vcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2])

        states.append(ep_states)
        actions.append(ep_actions)
        rewards.append(ep_rewards)
        total_steps += ep_length

        if use_qcritic:
            policy.apply_gradient_batch(states, actions, rewards, episode, qcritic, vcritic)
        else:
            policy.apply_gradient_batch(states, actions, rewards, episode, vcritic)

        policy_std = np.exp(policy.log_std.detach().numpy())
        #print(f"Policy std: {policy_std}")

        # Compute evaluation reward (last episode of batch).
        # total_reward = np.sum(rewards[-1])
        total_reward = np.mean([np.sum(ep) for ep in rewards])
        print("Episode: {0} | Average score in batch: {1}".format(episode, total_reward))

        writer.add_scalar("total_reward", total_reward, episode)
        #writer.add_scalar("policy_std", policy_std, episode)
        #writer.add_scalar("ep_length", ep_length, episode)

        if episode%checkpoint_freq == 0:
            target_critic = None
            critic = None
            if use_qcritic:
                critic = qcritic
                if use_target:
                    target_critic = target_qcritic
            save_path = "{}.tar".format(policy_type)
            save_checkpoint(policy, seed, env, config, use_qcritic, use_target, vcritic=vcritic, critic=critic, target_critic=target_critic, episode=episode, reward=total_reward, timesteps=total_steps, save_path=save_path)



if __name__ == '__main__':

    # TODO: add targetQ and soft update

    # TODO: 4000 episodes each, 25 iterations
    # TODO: Vary sample size for MC and Fixed Grid (1, 5, 10, 20, 100, 1000)
    # TODO: Simpsons 2000, 10 iterations (1, 5, 10, 20, 100, 1000)
    # TODO: Write metrics to file



    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True, type=str,
                    choices=['reinforce', 'mc', 'integrate'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_target', action='store_true')
    #parser.add_argument('--model_path', required=True, type=str)

    args = parser.parse_args()

    if args.seed <= 0:
        seed = random.randint(1,10000)
    else:
        seed = args.seed

    env = gym.make('InvertedPendulum-v1')
    print("Using seed {}".format(seed))

    config = Config()

    run(env, config, policy_type=args.policy, seed=seed, use_target=args.use_target)
