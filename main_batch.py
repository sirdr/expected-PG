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
env = gym.make('InvertedPendulum-v1')   # First env that worked.
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

for batch in range(10000):

    states = []
    actions = []
    rewards = []

    total_steps = 0

    while total_steps < 1000:

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
            # if(len(ep_actions) >= 2):
            #     qcritic.apply_gradient(ep_states[-2], ep_actions[-2], ep_rewards[-1], ep_states[-1], ep_actions[-1])
            #     # vcritic.apply_gradient(states[-2], actions[-2], rewards[-2], states[-1])

        states.append(ep_states)
        actions.append(ep_actions)
        rewards.append(ep_rewards)
        total_steps += ep_length

    policy.apply_gradient_batch(states, actions, rewards, qcritic, vcritic, batch)
    # qcritic.apply_gradient_batch(states, actions, rewards)
    # vcritic.apply_gradient_batch(states, rewards)
    # policy.apply_gradient_batch_critic(states, actions, rewards, qcritic, vcritic, batch)

    policy_std = np.exp(policy.log_std.detach().numpy())
    print(f"Policy std: {policy_std}")

    # Compute evaluation reward (last episode of batch).
    total_reward = np.sum(rewards[-1])
    print(f"Score of last episode in batch: {total_reward}")

    writer.add_scalar("total_reward", total_reward, batch)
    writer.add_scalar("policy_std", policy_std, batch)
    writer.add_scalar("ep_length", ep_length, batch)
