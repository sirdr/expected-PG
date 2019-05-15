import torch
import gym
import numpy as np
from policy import PolicyMC, PolicyReinforce, PolicyIntegrationTrapezoidal
from qcritic import QCritic
from vcritic import VCritic
from metrics import MetricsWriter
from utils import *
import random
import time
import os
import argparse
import config

# Logging stuff
from tensorboardX import SummaryWriter

def clip(x, min, max):
    return (x >= max) * max + (x <= min) * min + (x <= max * x >= min) * x

def get_config(env):
    if env == 'inv-pendulum':
        return config.InvertedPendulumConfig()
    elif env == "cheetah":
        return config.CheetahConfig()
    elif env == "walker":
        return config.WalkerConfig()
    elif env == "reacher":
        return config.ReacherConfig()
    elif env == "lander":
        return config.LanderConfig()
    else:
        print("Invalid environment name") #should never get here

def get_policy(policy_type, env, config, writer, num_actions):
    policy = None

    if policy_type == 'mc':
        policy = PolicyMC(env, config, writer)
    elif policy_type == 'reinforce':
        policy = PolicyReinforce(env, config, writer)
    elif policy_type == 'integrate':
        policy = PolicyIntegrationTrapezoidal(env, config, writer)
    else:
        print("invalid policy type") #should never get here
    return policy

def get_writer_name(policy_type, config, seed, use_target, env_name, num_actions, run_id='NA', exp_id='NA', evaluation=False, expected_sarsa=False):
    name = "{}-{}-{}-{}-exp_id={}-run_id={}-seed={}".format(policy_type, env_name, config.critic_lr, config.policy_lr, exp_id, run_id, seed)
    
    if policy_type == 'mc':
        name = name+'-num_samples={}'.format(num_actions)
    elif policy_type == 'reinforce':
        pass
    elif policy_type == 'integrate':
        name = name+'-num_actions={}'.format(num_actions)
    else:
        print("invalid policy type") #should never get here

    if config.clip_grad > 0:
        name = name + "-clip_grad={}".format(config.clip_grad)

    if config.clip_actions:
        name = name + "-clip_actions"
    else:
        name = name + "-tanh_actions"
    if expected_sarsa:
        name = name + "-expected_sarsa"
    if config.learn_std:
        name = name + "-learn_std"
    if use_target:
        name = name + "-use_target"
    if config.normalize_advantages:
        name = name + "-norm_adv"

    name = name + "-{}".format(int(time.time()))

    if evaluation:
        name = name+'-eval'
    return name

def save_checkpoint(policy, seed, env, config, use_qcritic, use_target, policy_type, env_name, num_actions,
                    run_id,
                    exp_id,
                    vcritic=None, 
                    critic=None, 
                    target_critic=None, 
                    reward=None, episode=None, 
                    timesteps=None, 
                    save_path='model.tar', 
                    verbose=True,
                    expected_sarsa=False):
    if verbose:
        print("Saving Training Checkpoint to {} ...".format(save_path))
    save_dict = {
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer_state_dict': policy.optimizer.state_dict(),
                'seed': seed,
                'env': env,
                'env_name': env_name,
                'policy_type': policy_type,
                'config': config,
                'use_qcritic': use_qcritic,
                'use_target': use_target,
                'run_id': run_id,
                'exp_id': exp_id,
                'num_actions': num_actions,
                'expected_sarsa': expected_sarsa
                }
    if vcritic is not None:
        save_dict['vcritic_state_dict'] = vcritic.state_dict()
        save_dict['vcritic_optimizer_state_dict'] = vcritic.optimizer.state_dict()
    if critic is not None:
        save_dict['critic_state_dict'] = critic.state_dict()
        save_dict['critic_optimizer_state_dict'] = critic.optimizer.state_dict()
    if reward is not None:
        save_dict['episode_reward'] = reward
    if episode is not None:
        save_dict['episode'] = episode
    if timesteps is not None:
        save_dict['timesteps'] = timesteps
    if target_critic is not None:
        save_dict['target_critic_state_dict'] = target_critic.state_dict()
        save_dict['target_critic_optimizer_state_dict'] = target_critic.optimizer.state_dict()

    torch.save(save_dict, save_path)
    if verbose:
        print("done")

def load_checkpoint(load_path, verbose=True):
    if verbose:
        print("Loading Training Checkpoint from {} ...".format(load_path))
    checkpoint = torch.load(load_path)

    if verbose:
        print("done")

    return checkpoint
