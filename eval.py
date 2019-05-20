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

from utils import *

# Logging stuff
from tensorboardX import SummaryWriter

def run_eval(env, metrics_writer, policy, num_episodes=1):

    print("Starting Policy Evaluation for {} Episode(s)".format(num_episodes))

    for ep in range(num_episodes):

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

        print("Episode: {} | Total Reward: {}".format(ep, total_reward))
        metrics_writer.write_metric(ep, "total_reward", total_reward)



def evaluate(load_path, num_episodes=1, record=False, record_dir='recordings'):

    checkpoint = load_checkpoint(load_path)
    seed = checkpoint['seed']
    config = checkpoint['config']
    env = checkpoint['env']
    use_qcritic = checkpoint['use_qcritic']
    use_target = checkpoint['use_target']
    policy_type = checkpoint['policy_type']
    env_name = checkpoint['env_name']
    num_actions = checkpoint['num_actions']
    expected_sarsa = checkpoint['expected_sarsa']
    run_id = checkpoint['run_id']
    exp_id = checkpoint['exp_id']

    if record:
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

    torch.manual_seed(seed)

    run_name = get_writer_name(policy_type, config, seed, use_target, env_name, num_actions, run_id=run_id, exp_id=exp_id, expected_sarsa=expected_sarsa, evaluation=True)
    metrics_writer = MetricsWriter(run_name)

    policy = get_policy(policy_type, env, config, metrics_writer, num_actions=num_actions, optimizer)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

    policy.eval()

    env = gym.make(env_name)

    if record:
        for i in range(record):
            model_name = load_path.split("/")[-1].split(".tar")[0]
            record_path = os.path.join(record_dir, model_name, "run-{}".format(i))
            env = gym.wrappers.Monitor(env,record_path, video_callable=lambda x: True, resume=True)
            run_eval(env, metrics_writer, policy, num_episodes=1)

    else:
        run_eval(env, metrics_writer, policy, vcritic, qcritic, num_episodes=num_episodes)




if __name__ == '__main__':

    # TODO: add convergence criterion
    # or fixed number of episodes ~ 4000
    # TODO: add targetQ and soft update
    # TODO: add the fixed grid

    # TODO: add checkpointing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--record', type=int, default=0)

    args = parser.parse_args()

    num_episodes = max(args.num_episodes, 1)
    record = max(args.record, 0)


    evaluate(args.model_path, num_episodes=num_episodes, record=record)
