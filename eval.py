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



def evaluate(load_path, outdir="", num_episodes=1, record=False, record_dir='recordings'):

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

    runs_dir = os.path.join(outdir)

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir, exist_ok=True)

    #run_name = get_writer_name(policy_type, config, seed, use_target, env_name, num_actions, run_id=run_id, exp_id=exp_id, expected_sarsa=expected_sarsa, evaluation=True)
    run_name = load_path.split("/")[-1] + "-eval"
    metrics_writer = MetricsWriter(run_name, runs_dir=runs_dir, runs_only=True)

    policy = get_policy(policy_type, env, config, metrics_writer, num_actions=num_actions)
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
        run_eval(env, metrics_writer, policy, num_episodes=num_episodes)




if __name__ == '__main__':

    # TODO: add convergence criterion
    # or fixed number of episodes ~ 4000
    # TODO: add targetQ and soft update
    # TODO: add the fixed grid

    # TODO: add checkpointing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--checkpoint_episode', type=int, default=4999)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--record', type=int, default=0)
    parser.add_argument('--outdir', type=str, default="")

    args = parser.parse_args()

    num_episodes = max(args.num_episodes, 1)
    record = max(args.record, 0)

    if os.path.isdir(args.model_path):
        files = os.listdir(args.model_path)
        files = [os.path.join(args.model_path, f) for f in files]
        if args.outdir == "":
            outdir = args.model_path
        else:
            outdir = args.outdir
    else:
        files = [args.model_path]
        outdir = args.outdir

    for file in files:

        if "episode={}".format(args.checkpoint_episode) in file:
            print("Evaluating {}".format(file)) 
            evaluate(file, outdir=outdir, num_episodes=num_episodes, record=record)
