import numpy as np
import gym
import torch
import random
import os
import argparse
import time

from policy import PolicyMC, PolicyReinforce, PolicyIntegrationTrapezoidal
from qcritic import QCritic
from vcritic import VCritic
from config import Config
from metrics import MetricsWriter
from utils import *

def get_env_name(proxy_name):
    if proxy_name == 'inv-pendulum':
        return 'InvertedPendulum-v1'
    if proxy_name == 'walker':
        return 'Walker2d-v1'
    if proxy_name == 'cheetah':
        return 'HalfCheetah-v1'
    if proxy_name == 'reacher':
        return 'Reacher2d-v1'


def run(env, config,
        policy_type='integrate',
        seed=7,
        use_target=False,
        use_gpu=False,
        checkpoint_freq=1000,
        num_episodes=4000):

    env.seed(seed)
    torch.manual_seed(seed)

    run_name = get_writer_name(policy_type, config, seed)
    metrics_writer = MetricsWriter(run_name)

    vcritic = VCritic(env, config)
    policy = get_policy(policy_type, env, config, metrics_writer)

    if policy_type == 'integrate' or policy_type == 'mc':
        use_qcritic = True
        target_qcritic = None
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

    total_steps = 0

    for episode in range(num_episodes):

        observation = env.reset()
        done = False
        ep_length = 0

        ep_states = [observation]
        ep_actions = []
        ep_rewards = []

        while not done:
            # env.render()
            action = policy.get_action(observation)
            observation, reward, done, info = env.step(action)
            ep_states.append(observation)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_length += 1
            if(len(ep_actions) >= 2):
                if use_qcritic:
                    qcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2], ep_actions[-1], target_q=target_qcritic)
                    if use_target:
                        q_state_dict = qcritic.state_dict()
                        # target_qcritic.load_state_dict(q_state_dict)
                        target_state_dict = target_qcritic.state_dict()
                        for name, param in target_state_dict.items():
                            if not ("weight" in name or "bias" in name):
                                continue
                            param.data = (1-config.tau)*param.data + config.tau*q_state_dict[name].data
                            target_state_dict[name].copy_(param)
                        target_qcritic.load_state_dict(target_state_dict)
                        
                vcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2])

        if use_qcritic:
            qcritic.apply_gradient(ep_states[-2], ep_actions[-1], ep_rewards[-1], None, None, target_q = target_qcritic)
        vcritic.apply_gradient(ep_states[-2], ep_actions[-1], ep_rewards[-1], None)

        if use_qcritic:
            policy.apply_gradient_episode(ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic)
        else:
            policy.apply_gradient_episode(ep_states, ep_actions, ep_rewards, episode, vcritic)

        total_reward = np.sum(ep_rewards)
        print("Episode: {0} | Average score in batch: {1}".format(episode, total_reward))
        metrics_writer.write_metric(episode, "total_reward", total_reward)

        if episode % checkpoint_freq == 0:
            target_critic = None
            critic = None
            if use_qcritic:
                critic = qcritic
                if use_target:
                    target_critic = target_qcritic
            outdir = "checkpoints/"
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            save_path = os.path.join(outdir, "{}.tar".format(run_name))
            save_checkpoint(policy, seed, env, config, use_qcritic, use_target, vcritic=vcritic, critic=critic, target_critic=target_critic, episode=episode, reward=total_reward, timesteps=total_steps, save_path=save_path)



if __name__ == '__main__':

    # TODO: 4000 episodes each, 25 iterations
    # TODO: Vary sample size for MC and Fixed Grid (1, 5, 10, 20, 100, 1000)
    # TODO: Simpsons 2000, 10 iterations (1, 5, 10, 20, 100, 1000)

    ## TODO: figure out how to run using GPU
    ## TODO: add other envs / make sure that trapezoidal works on higher dimensions

    ## TODO: add the done
    ## TODO: figure out the detach issue / target Q
    ## TODO: investigate unlearning

    ## TODO: run integrate to investigate unlearning

    ## TODO: add gradient comparison script


    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True, type=str,
                    choices=['reinforce', 'mc', 'integrate'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_target', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--env', required=True, type=str,
                        choices=['inv-pendulum', 'walker', 'cheetah', 'reacher'])
    #parser.add_argument('--model_path', required=True, type=str)


    args = parser.parse_args()

    if args.seed <= 0:
        seed = random.randint(1,10000)
    else:
        seed = args.seed

    env_name = get_env_name(args.env)
    env = gym.make(env_name)
    print("Using seed {}".format(seed))

    config = Config()
    run(env, config, policy_type=args.policy, seed=seed, use_target=args.use_target, use_gpu=args.use_gpu)
