import gym
import numpy as np
import torch
from policy import Policy
from qcritic import QCritic
from vcritic import VCritic
from config import Config

# Logging stuff
from tensorboardX import SummaryWriter

# env = gym.make('MountainCarContinuous-v0')
env = gym.make('Pendulum-v0')
env.seed(1); torch.manual_seed(1);

config = Config()
policy = Policy(env, config)
qcritic = QCritic(env, config)
vcritic = VCritic(env, config)

score_writer = SummaryWriter('score')

for ep in range(1000):

    observation = env.reset()
    done = False

    states = [observation]
    actions = []
    rewards = []

    ep_length = 0

    while not done:# and ep_length < 100:
        # env.render()
        action = policy.get_action(observation)
        action_value = qcritic(torch.from_numpy(observation).float(), torch.from_numpy(action).float())
        state_value = vcritic(torch.from_numpy(observation).float())
        print("Action value: {} - State value: {} - Advantage: {}".format(action_value, state_value, action_value - state_value))
        observation, reward, done, info = env.step(action)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        policy.apply_gradient2(observation, qcritic, vcritic, ep_length)
        ep_length += 1
        if(len(actions) >= 2):
            qcritic.apply_gradient(states[-2], actions[-2], rewards[-1], states[-1], actions[-1])
            vcritic.apply_gradient(states[-2], actions[-2], rewards[-1], states[-1])

    total_reward = np.sum(rewards)
    print("Total reward for episode: {}".format(total_reward))
    score_writer.add_scalar("total_reward", total_reward, ep)

    # Apply gradient for specified states. Uses critic to compute policy gradient.
    # policy.apply_gradient(states, critic, ep)
