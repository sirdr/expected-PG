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
    id_2_seed = {1 : 5789 ,
                2 : 9584,
                3 : 2494,
                4 : 4370,
                5 : 7664,
                6 : 6659,
                7 : 2619,
                8 : 5460,
                9 : 8694,
                10 : 150,
                11 : 1857,
                12 : 4640,
                13 : 4956,
                14 : 2649,
                15 : 1667}
    seed = id_2_seed[run_id]
    return seed

def soft_update(target_model, model, tau=0.):
    model_state_dict = model.state_dict()
    target_state_dict = target_model.state_dict()
    for name, param in target_state_dict.items():
        if not ("weight" in name or "bias" in name):
            # print(f"Not soft updating param {name}")
            continue
        # pdb.set_trace()
        # param.data = tau*param.data + (1-tau)*model_state_dict[name].data
        # target_state_dict[name].copy_(param)
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
        return 'Reacher2d-v1'

def run(env_name, config,
        policy_type='integrate',
        seed=7,
        use_target=False,
        use_policy_target=False,
        use_gpu=False,
        checkpoint_freq=1000,
        num_episodes=5000,
        run_id='NA',
        exp_id='NA'):

    env = gym.make(env_name)
    print("Using seed {}".format(seed))

    env.seed(seed)
    torch.manual_seed(seed)

    num_actions = config.n_samples_per_state
    run_name = get_writer_name(policy_type, config, seed, use_target, env_name, num_actions, run_id=run_id, exp_id=exp_id)
    metrics_writer = MetricsWriter(run_name)

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
    timesteps = 0

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
                vcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2])
                if use_qcritic:
                    if use_policy_target:
                        next_action = target_policy.get_action(observation)
                    else:
                        next_action = ep_actions[-1]
                    qcritic.apply_gradient(ep_states[-3], ep_actions[-2], ep_rewards[-2], ep_states[-2], next_action, target_q=target_qcritic)
                    if use_target:
                        soft_update(target_qcritic, qcritic, tau=config.tau)
                    if use_policy_target:
                        soft_update(target_policy, policy, tau=config.tau)

        vcritic.apply_gradient(ep_states[-2], ep_actions[-1], ep_rewards[-1], None)
        if use_qcritic:
            qcritic.apply_gradient(ep_states[-2], ep_actions[-1], ep_rewards[-1], None, None, target_q = target_qcritic)

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

    # TODO: 7000 episodes each, ~10 iterations
    # TODO: Vary sample size for MC and Fixed Grid (1, 5, 10, 20, 100, 1000)
    # TODO: Simpsons 2000, 10 iterations (1, 5, 10, 20, 100, 1000)

    ## TODO: figure out how to run using GPU
    ## TODO: add other envs / make sure that trapezoidal works on higher dimensions

    ## TODO: investigate unlearning

    ## TODO: run integrate to investigate unlearning
    ## TODO: add gradient comparison script
    ## TODO: add env_name, task_id to writer

    ## TODO: finish eval 

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True, type=str,
                    choices=['reinforce', 'mc', 'integrate'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_target', action='store_true')
    parser.add_argument('--use_policy_target', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--env', required=True, type=str,
                        choices=['inv-pendulum', 'walker', 'cheetah', 'reacher'])
    parser.add_argument('--run_id', type=str, default='NA')
    parser.add_argument('--exp_id', type=str, default='NA')
    parser.add_argument('--num_actions', type=int, default=100)
    #parser.add_argument('--model_path', required=True, type=str)


    args = parser.parse_args()

    if args.run_id != 'NA':
        seed = run_id_2_seed(args.run_id)
    else:
        seed = args.seed

    env_name = get_env_name(args.env)

    config = get_config(args.env)
    config.n_samples_per_state = args.num_actions

    run(env_name, config,
        policy_type=args.policy,
        seed=seed,
        use_target=args.use_target,
        use_policy_target=args.use_policy_target,
        use_gpu=args.use_gpu,
        run_id=args.run_id,
        exp_id=args.exp_id)
