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

if __name__ == '__main__':

    # TODO: add convergence criterion
    # or fixed number of episodes ~ 4000
    # TODO: add targetQ and soft update
    # TODO: add the fixed grid

    # TODO: add checkpointing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)

    args = parser.parse_args()


    file = args.model_path

    checkpoint = load_checkpoint(file)
    config = checkpoint['config']

    print("policy_layers: {}".format(config.policy_layers))
    print("qcritic_layers: {}".format(config.qcritic_layers))
    print("vcritic_layers: {}".format(config.vcritic_layers))
    print("gamma: {}".format(config.gamma))
    print("eps: {}".format(config.eps))
    print("gamma: {}".format(config.gamma))
    print("critic_lr: {}".format(config.critic_lr))
    print("policy_lr: {}".format(config.policy_lr))
    print("policy_lr_decay: {}".format(config.policy_lr_decay))
    print("critic_lr_decay: {}".format(config.critic_lr_decay))
    print("policy_lr_step_size: {}".format(config.policy_lr_step_size))
    print("critic_lr_step_size: {}".format(config.critic_lr_step_size))
    print("n_samples_per_state: {}".format(config.n_samples_per_state))
    print("normalize_advantages: {}".format(config.normalize_advantages))
    print("learn_std: {}".format(config.learn_std))
    print("tau: {}".format(config.tau))
    print("action_std: {}".format(config.action_std))
    print("clip_actions: {}".format(config.clip_actions))
    print("clip_grad: {}".format(config.clip_grad))
    print("num_episodes: {}".format(config.num_episodes))
    print("max_steps: {}".format(config.max_steps))
    print("clever: {}".format(config.clever))


    
