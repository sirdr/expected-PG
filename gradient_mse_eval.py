import gym
import numpy as np
import torch
from policy import PolicyMC, PolicyReinforce, PolicyIntegrationTrapezoidal
from qcritic import QCritic
from vcritic import VCritic
import random
import time
import os
import argparse
import pdb

from utils import *

# Logging stuff
from tensorboardX import SummaryWriter

def eval_gradients(env, policy_reinforce, mc_policies, qcritic, vcritic, num_episodes=1000):

    print("Starting Policy Evaluation for {} Episode(s)".format(num_episodes))

    reinforce_gradients = []

    for ep in range(num_episodes):

        observation = env.reset()
        done = False

        ep_states, ep_actions, ep_rewards = [observation], [], []
        while not done:
            action = policy_reinforce.get_action(observation)
            observation, reward, done, info = env.step(action)
            ep_states.append(observation); ep_actions.append(action); ep_rewards.append(reward)
        reinforce_gradients.append(policy_reinforce.compute_gradient_episode(ep_states, ep_actions, ep_rewards, vcritic).clone())

    reinforce_gradients = torch.stack(reinforce_gradients)
    true_gradient = torch.mean(reinforce_gradients, dim=0)

    mse_dict = {}
    mse_dict['reinforce'] = torch.mean(torch.norm(reinforce_gradients - true_gradient, dim=(1,2))**2).data

    for num_actions in mc_policies:
        print(f"{num_actions} sample MC.")
        policy = mc_policies[num_actions]
        gradients = []

        for ep in range(num_episodes):

            observation = env.reset()
            done = False

            ep_states, ep_actions, ep_rewards = [observation], [], []
            while not done:
                action = policy_reinforce.get_action(observation)
                observation, reward, done, info = env.step(action)
                ep_states.append(observation); ep_actions.append(action); ep_rewards.append(reward)
            gradients.append(policy.compute_gradient_episode(ep_states, ep_actions, ep_rewards, qcritic, vcritic).clone())

        gradients = torch.stack(gradients)
        mse_dict[f'mc-{num_actions}'] = torch.mean(torch.norm(gradients - true_gradient, dim=(1,2))**2).data

    print(mse_dict)

def evaluate(load_path, num_episodes=1, record=False, record_dir='recordings'):

    checkpoint = load_checkpoint(load_path)
    seed = checkpoint['seed']
    config = checkpoint['config']
    env = checkpoint['env']
    use_qcritic = checkpoint['use_qcritic']
    env_name = checkpoint['env_name']
    num_actions = checkpoint['num_actions']
    expected_sarsa = checkpoint['expected_sarsa']
    run_id = checkpoint['run_id']
    exp_id = checkpoint['exp_id']

    if record:
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

    torch.manual_seed(seed)

    metrics_writer = MetricsWriter("lambda")

    policy_reinforce = PolicyReinforce(env, config, metrics_writer)
    policy_reinforce.load_state_dict(checkpoint['policy_state_dict'])
    policy_reinforce.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    policy_reinforce.eval()

    mc_policies = {}
    #for num_actions in [1,2,4,8,16,32,64,128,256,512]:
    for num_actions in [256]:
        policy = PolicyMC(env, config, metrics_writer, num_actions=num_actions)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        policy.eval()
        mc_policies[num_actions] = policy

    vcritic = VCritic(env, config)
    vcritic.load_state_dict(checkpoint['vcritic_state_dict'])
    vcritic.optimizer.load_state_dict(checkpoint['vcritic_optimizer_state_dict'])
    qcritic = QCritic(env, config, metrics_writer)
    qcritic.load_state_dict(checkpoint['critic_state_dict'])
    qcritic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    env = gym.make(env_name)

    eval_gradients(env, policy_reinforce, mc_policies, qcritic, vcritic, num_episodes=num_episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--record', type=int, default=0)

    args = parser.parse_args()

    evaluate(args.model_path, num_episodes=1000)
