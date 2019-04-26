import gym
import numpy as np
import torch
from policy_backward import PolicyBackward
from vcritic import VCritic
from config import Config
import random
import time

# Logging stuff
from tensorboardX import SummaryWriter

env = gym.make('InvertedPendulum-v1')
seed = random.randint(1,10000)
env.seed(seed); torch.manual_seed(seed);

config = Config()

writer = SummaryWriter(f'score/backward-{config.critic_lr}-{config.policy_lr}-{config.normalize_advantages}-BS{config.batch_size}-learnStd={config.learn_std}-{seed}-{int(time.time())}')

vcritic = VCritic(env, config)
policy = PolicyBackward(env, config, writer)

policy.train()

for batch in range(4000):

    states = []
    actions = []
    rewards = []

    total_steps = 0

    # while total_steps < config.batch_size:

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

    states.append(ep_states)
    actions.append(ep_actions)
    rewards.append(ep_rewards)
    total_steps += ep_length

    policy.apply_gradient_batch(states, actions, rewards, batch, vcritic)

    policy_std = np.exp(policy.log_std.detach().numpy())
    print(f"Policy std: {policy_std}")

    # Compute evaluation reward (last episode of batch).
    # total_reward = np.sum(rewards[-1])
    total_reward = np.mean([np.sum(ep) for ep in rewards])
    print(f"Average score in batch: {total_reward}")

    writer.add_scalar("total_reward", total_reward, batch)
    writer.add_scalar("policy_std", policy_std, batch)
    writer.add_scalar("ep_length", ep_length, batch)
