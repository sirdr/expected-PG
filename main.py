import gym
import numpy as np
import torch
from policy_reinforce import PolicyReinforce
from policy_backward import PolicyBackward
from policy_integration import PolicyIntegration
from qcritic import QCritic
from vcritic import VCritic
from config import Config

# Logging stuff
from tensorboardX import SummaryWriter

env = gym.make('InvertedPendulum-v1')
env.seed(1); torch.manual_seed(1);

writer = SummaryWriter('score/backward-baseline')

config = Config()
qcritic = QCritic(env, config)
# vcritic = VCritic(env, config)
vcritic = None

# policy = PolicyReinforce(env, config, writer)
# policy = PolicyBackward(env, config, writer)
policy = PolicyIntegration(env, config, writer)

policy.train()
qcritic.train()
# vcritic.train()

for batch in range(10000):

    states = []
    actions = []
    rewards = []

    total_steps = 0

    while total_steps < config.batch_size:

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
                qcritic.apply_gradient(ep_states[-2], ep_actions[-2], ep_rewards[-1], ep_states[-1], ep_actions[-1])
                # vcritic.apply_gradient(ep_states[-2], ep_actions[-2], ep_rewards[-1], ep_states[-1])

        states.append(ep_states)
        actions.append(ep_actions)
        rewards.append(ep_rewards)
        total_steps += ep_length

    # policy.apply_gradient_batch(states, actions, rewards, batch, vcritic=vcritic)
    policy.apply_gradient_batch(states, actions, rewards, batch, qcritic, vcritic=vcritic)

    # policy_std = np.exp(policy.log_std.detach().numpy())
    # print(f"Policy std: {policy_std}")

    # Compute evaluation reward (last episode of batch).
    total_reward = np.mean([np.sum(ep) for ep in rewards])
    print(f"Score of last episode in batch: {total_reward}")

    writer.add_scalar("total_reward", total_reward, batch)
    # writer.add_scalar("policy_std", policy_std, batch)
    writer.add_scalar("ep_length", ep_length, batch)
