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
# env = gym.make('Pendulum-v0')
env = gym.make('InvertedPendulum-v1')
# env = gym.make('HalfCheetah-v1')
env.seed(1); torch.manual_seed(1);

writer = SummaryWriter('score')

config = Config()
policy = Policy(env, config, writer)
qcritic = QCritic(env, config)
vcritic = VCritic(env, config)

policy.train()
qcritic.train()
vcritic.train()

for ep in range(25000):

    observation = env.reset()
    done = False

    states = [observation]
    actions = []
    rewards = []

    ep_length = 0

    while not done:# and ep_length < 100:
        env.render()
        action = policy.get_action(observation)
        action_value = qcritic(torch.from_numpy(observation).float(), torch.from_numpy(action).float())
        state_value = vcritic(torch.from_numpy(observation).float())
        print("Action: {} - Action value: {} - State value: {} - Advantage: {}".format(action, action_value, state_value, action_value - state_value))
        observation, reward, done, info = env.step(action)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        ep_length += 1
        # if(len(actions) >= 2):
        #     qcritic.apply_gradient(states[-2], actions[-2], rewards[-2], states[-1], actions[-1])
            # vcritic.apply_gradient(states[-2], actions[-2], rewards[-2], states[-1])

    policy.apply_gradient5(states[:-1], actions, rewards, qcritic, vcritic)

    total_reward = np.sum(rewards)

    writer.add_scalar("total_reward", total_reward, ep)
    writer.add_scalar("ep_length", ep_length, ep)
