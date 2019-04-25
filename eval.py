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

def evaluate(load_path):

    checkpoint = load_checkpoint(load_path)
    seed = checkpoint['seed']
    config = checkpoint['config']
    #policy_type = checkpoint['policy_type']
    env = checkpoint['env']
    use_qcritic = checkpoint['use_qcritic']
    use_target = checkpoint['use_target']

    env.seed(seed) 
    torch.manual_seed(seed)

    writer = get_writer('integrate', config, seed)

    vcritic = VCritic(env, config)
    vcritic.load_state_dict(checkpoint['vcritic_state_dict'])
    vcritic.optimizer.load_state_dict(checkpoint['vcritic_optimizer_state_dict'])

    policy = get_policy('integrate', env, config, writer)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])


    if use_qcritic:
        qcritic = QCritic(env, config)
        qcritic.load_state_dict(checkpoint['critic_state_dict'])
        qcritic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        qcritic.eval()
        if use_target:
            target_qcritic = QCritic(env, config)
            target_qcritic.load_state_dict(checkpoint['target_critic_state_dict'])
            target_qcritic.optimizer.load_state_dict(checkpoint['target_critic_optimizer_state_dict'])
            target_qcritic.eval()

    policy.eval()
    vcritic.eval()

    observation = env.reset()
    done = False
    ep_length = 0

    ep_rewards = []

    while not done:
        # env.render()
        action = policy.get_action(observation)
        observation, reward, done, info = env.step(action)
        ep_rewards.append(reward)
        ep_length += 1

    total_reward = np.sum(ep_rewards)

    print("Total Evaluation Reward: {}".format(total_reward))



if __name__ == '__main__':

    # TODO: add convergence criterion
    # or fixed number of episodes ~ 4000
    # TODO: add targetQ and soft update
    # TODO: add the fixed grid

    # TODO: add checkpointing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    args = parser.parse_args()

    evaluate(args.model_path)
