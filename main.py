import numpy as np
import gym
import torch
import random
import os
import argparse
import time
import pdb

from policy import PolicyMC, PolicyReinforce, PolicyIntegrationTrapezoidal
from qcritic import QCritic
from vcritic import VCritic
from metrics import MetricsWriter
from utils import *

def run_id_2_seed(run_id):
    id_2_seed = {'1' : 5789 ,
                '2' : 9584,
                '3' : 2494,
                '4' : 4370,
                '5' : 7664,
                '6' : 6659,
                '7' : 2619,
                '8' : 5460,
                '9' : 8694,
                '10' : 150,
                '11' : 1857,
                '12' : 4640,
                '13' : 4956,
                '14' : 2649,
                '15' : 1667}
    seed = id_2_seed[run_id]
    return seed

def soft_update(target_model, model, tau=0.):
    model_state_dict = model.state_dict()
    target_state_dict = target_model.state_dict()
    for name, param in target_state_dict.items():
        if not ("weight" in name or "bias" in name):
            continue
        transformed_param = tau*param + (1-tau)*model_state_dict[name]
        target_state_dict[name].copy_(transformed_param)
    target_model.load_state_dict(target_state_dict)


def get_env_name(proxy_name):
    if proxy_name == 'inv-pendulum':
        return 'InvertedPendulum-v1'
    if proxy_name == 'walker':
        return 'Walker2d-v1'
    if proxy_name == 'cheetah':
        return 'HalfCheetah-v1'
    if proxy_name == 'reacher':
        return 'Reacher-v1'
    if proxy_name == 'lander':
        return 'LunarLanderContinuous-v2'

def run(env_name, config,
        policy_type='integrate',
        seed=7,
        use_target=False,
        use_policy_target=False,
        use_gpu=False,
        checkpoint_freq=1000,
        num_episodes=5000,
        run_id='NA',
        exp_id='NA',
        results_dir='',
        expected_sarsa=False):

    checkpoint_dir = os.path.join(results_dir, 'checkpoints/')
    runs_dir = os.path.join(results_dir, 'runs/')
    score_dir = os.path.join(results_dir, 'score/')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    env = gym.make(env_name)
    print("Using seed {}".format(seed))

    env.seed(seed)
    torch.manual_seed(seed)

    num_actions = config.n_samples_per_state
    run_name = get_writer_name(policy_type, config, seed, use_target, env_name, num_actions, run_id=run_id, exp_id=exp_id, expected_sarsa=expected_sarsa)
    metrics_writer = MetricsWriter(run_name, runs_dir=runs_dir, score_dir=score_dir)

    vcritic = VCritic(env, config)
    policy = get_policy(policy_type, env, config, metrics_writer, num_actions)

    if policy_type == 'integrate' or policy_type == 'mc':
        use_qcritic = True
        target_qcritic = None
        qcritic = QCritic(env, config, metrics_writer)
        qcritic.train()
        if use_target:
            target_qcritic = QCritic(env, config, metrics_writer)
            target_qcritic.load_state_dict(qcritic.state_dict())
            target_qcritic.eval()
        if use_policy_target:
            target_policy = get_policy(policy_type, env, config, metrics_writer)
            target_policy.load_state_dict(policy.state_dict())
            target_policy.eval()

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
            # if episode%50 == 0:
            #     env.render()
            action = policy.get_action(observation)
            observation, reward, done, info = env.step(action)
            total_steps += 1
            ep_states.append(observation)
            ep_actions.append(action)
            ep_rewards.append(reward)

            ep_length += 1
            if(len(ep_actions) >= 2):
                if use_qcritic:
                    if use_policy_target:
                        next_action = target_policy.get_action(observation)
                    else:
                        next_action = ep_actions[-1]
                    if expected_sarsa:
                        qcritic.apply_gradient_expected(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2], policy, last_state = False, target_q=target_qcritic)
                    else:
                        qcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2], next_action, target_q=target_qcritic)

                    if use_target:
                        soft_update(target_qcritic, qcritic, tau=config.tau)
                    if use_policy_target:
                        soft_update(target_policy, policy, tau=config.tau)

        vcritic.apply_gradient_episode(ep_states, ep_rewards)
        if use_qcritic:
            if expected_sarsa:
                qcritic.apply_gradient_expected(ep_states[-2], ep_actions[-1], ep_rewards[-1], ep_states[-1], policy, last_state = True, target_q=target_qcritic)
            else:
                qcritic.apply_gradient(ep_states[-2], ep_actions[-1], ep_rewards[-1], None, None, target_q=target_qcritic)

        if use_qcritic:
            policy.apply_gradient_episode(ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic)
        else:
            policy.apply_gradient_episode(ep_states, ep_actions, ep_rewards, episode, vcritic)

        total_reward = np.sum(ep_rewards)
        print("Episode: {0} | Average score in batch: {1}".format(episode, total_reward))
        metrics_writer.write_metric(episode, "total_reward", total_reward)
        metrics_writer.write_metric(total_steps, "total_reward_by_steps", total_reward)
        metrics_writer.write_metric(episode, "policy_std", torch.exp(policy.log_std)[0])

        if episode % checkpoint_freq == 0:
            target_critic = None
            critic = None
            if use_qcritic:
                critic = qcritic
                if use_target:
                    target_critic = target_qcritic
            save_path = os.path.join(checkpoint_dir, "{}.tar".format(run_name))
            save_checkpoint(policy, seed, env, config, use_qcritic, use_target, policy_type, env_name, num_actions,
                            run_id,
                            exp_id,
                            vcritic=vcritic,
                            critic=critic,
                            target_critic=target_critic,
                            episode=episode,
                            reward=total_reward,
                            timesteps=total_steps,
                            save_path=save_path,
                            expected_sarsa=expected_sarsa)

    target_critic = None
    critic = None
    if use_qcritic:
        critic = qcritic
        if use_target:
            target_critic = target_qcritic
    save_path = os.path.join(checkpoint_dir, "{}.tar".format(run_name))
    save_checkpoint(policy, seed, env, config, use_qcritic, use_target, policy_type, env_name, num_actions,
                    run_id,
                    exp_id,
                    vcritic=vcritic,
                    critic=critic,
                    target_critic=target_critic,
                    episode=episode,
                    reward=total_reward,
                    timesteps=total_steps,
                    save_path=save_path,
                    expected_sarsa=expected_sarsa)

if __name__ == '__main__':

    # TODO: 7000 episodes each, ~10 iterations
    # TODO: Vary sample size for MC and Fixed Grid (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    # TODO: Simpsons 2000, 10 iterations (1, 5, 10, 20, 100, 1000)

    # TODO: figure out how to run using GPU

    # TODO: run integrate to investigate unlearning
    # TODO: add gradient comparison script

    # TODO: keep track of num samples seen
    # TODO: keep track of timesteps
    # TODO: normalize rewards

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True, type=str,
                    choices=['reinforce', 'mc', 'integrate'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_target', action='store_true')
    parser.add_argument('--use_policy_target', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--expected_sarsa', action='store_true')
    parser.add_argument('--clip_grad', type=int, default=0)
    parser.add_argument('--clip_actions', action='store_true')
    parser.add_argument('--env', required=True, type=str,
                        choices=['inv-pendulum', 'walker', 'cheetah', 'reacher', 'lander'])
    parser.add_argument('--run_id', type=str, default='NA')
    parser.add_argument('--exp_id', type=str, default='NA')
    parser.add_argument('--num_actions', type=int, default=100)
    parser.add_argument('--num_episodes', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='')
    #parser.add_argument('--model_path', required=True, type=str)


    args = parser.parse_args()

    if args.run_id != 'NA':
        seed = run_id_2_seed(args.run_id)
    else:
        seed = args.seed

    env_name = get_env_name(args.env)

    config = get_config(args.env)
    config.n_samples_per_state = args.num_actions
    config.clip_actions = args.clip_actions
    config.clip_grad = args.clip_grad

    if args.num_episodes > 0:
        num_episodes = args.num_episodes
        config.num_episodes = args.num_episodes
    else:
        num_episodes = config.num_episodes

    start_time = time.time()

    run(env_name, config,
        policy_type=args.policy,
        seed=seed,
        use_target=args.use_target,
        use_policy_target=args.use_policy_target,
        use_gpu=args.use_gpu,
        run_id=args.run_id,
        exp_id=args.exp_id,
        num_episodes=num_episodes,
        results_dir=args.results_dir,
        expected_sarsa=args.expected_sarsa)

    end_time = time.time()

    print("start time: {} | end time: {} | duration: {}".format(start_time, end_time, end_time-start_time))
