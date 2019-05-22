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

    print(vars(config))

    
