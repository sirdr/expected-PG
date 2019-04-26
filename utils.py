import torch
import gym
import numpy as np
from policy import PolicyMC, PolicyReinforce, PolicyIntegrationTrapezoidal
from qcritic import QCritic
from vcritic import VCritic
from config import Config
import random
import time
import os
import argparse

# Logging stuff
from tensorboardX import SummaryWriter

def clip(x, min, max):
    return (x >= max) * max + (x <= min) * min + (x <= max * x >= min) * x

def get_policy(policy_type, env, config, writer):
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

def get_writer_name(policy_type, config, seed):
    name = ""
    if policy_type == 'mc':
        name = f'mc-{config.critic_lr}-{config.policy_lr}-{config.normalize_advantages}-SS{config.n_samples_per_state}-learnStd={config.learn_std}-{seed}-{int(time.time())}'
    elif policy_type == 'reinforce':
        name = f'reinforce-{config.critic_lr}-{config.policy_lr}-{config.normalize_advantages}-learnStd={config.learn_std}-{seed}-{int(time.time())}'
    elif policy_type == 'integrate':
        name = f'integration-{config.critic_lr}-{config.policy_lr}-{config.normalize_advantages}-learnStd={config.learn_std}-{seed}-{int(time.time())}'
    else:
        print("invalid policy type") #should never get here
    return name

def save_checkpoint(policy, seed, env, config, use_qcritic, use_target, vcritic=None, critic=None, target_critic=None, reward=None, episode=None, timesteps=None, save_path='model.tar', verbose=True):
    if verbose:
        print("Saving Training Checkpoint to {} ...".format(save_path))
    save_dict = {
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer_state_dict': policy.optimizer.state_dict(),
                'seed': seed,
                'env': env,
                'config': config,
                'use_qcritic': use_qcritic,
                'use_target': use_target
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
