import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
import pdb
import integration
import random

'''
    Policy is trained using any policy gradient method.
'''
class Policy(nn.Module):
    def __init__(self, env, config, writer):
        super(Policy, self).__init__()
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_space_high = torch.from_numpy(env.action_space.high)
        self.action_space_low = torch.from_numpy(env.action_space.low)

        self.l1 = nn.Linear(self.state_space, 16)
        self.l2 = nn.Linear(16, self.action_space)
        self.log_std = torch.nn.Parameter(torch.tensor([np.log(0.1)], dtype=torch.float32), requires_grad=True)

        self.gamma = config.gamma
        self.sigma = config.sigma

        self.optimizer = optim.Adam(self.parameters(), lr=config.policy_lr)

        # Episode policy and reward history
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

        self.writer = writer

    '''
        Estimate the mean of a Gaussian (continuous) stochastic policy.
    '''
    def forward(self, state):
        out = self.l1(state)
        out = F.relu(out)
        out = self.l2(out)
        return out

    def get_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_mean = self.forward(state)
        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_mean, std)
        sample = dist.sample().numpy()
        return np.clip(sample, self.action_space_low.numpy(), self.action_space_high.numpy())

    def compute_advantages(self, rewards_by_path, normalize=True):
        advantages = []
        for reward_path in rewards_by_path:
            g = np.array(reward_path)
            n_transitions = len(g)
            g = (self.gamma ** np.arange(n_transitions)) * g
            g = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
            advantages.append(g)
        advantages = np.concatenate(advantages)
        if normalize:
            advantages = (advantages - np.mean(advantages)) / np.std(advantages)
        return advantages

    '''
    Batch version - no critic, only sampling.
    '''
    def apply_gradient_batch(self, states, actions, rewards, qcritic, vcritic, batch):

        self.optimizer.zero_grad()
        grads = {}
        for name, param in self.named_parameters():
            grads[name] = torch.zeros_like(param)

        n_episodes = len(states)
        states = [state for episode in states for state in episode[:-1]]
        actions = [action for episode in actions for action in episode]
        n_actions = len(actions)
        advantages = self.compute_advantages(rewards, normalize=True)

        print(f"Actions in batch: {n_actions}")

        for i in range(n_actions):

            state = states[i]
            action = actions[i]
            advantage = advantages[i]

            # std = torch.exp(self.log_std)
            # dist = torch.distributions.normal.Normal(self.forward(torch.from_numpy(state).float()), std)
            # (-dist.log_prob(torch.from_numpy(action).float()) * advantages[i]).backward()

            # print(list(self.parameters())[0].grad)

            for name, param in self.named_parameters():
                std = torch.exp(self.log_std)
                dist = torch.distributions.normal.Normal(self.forward(torch.from_numpy(state).float()), std)
                grads[name] -= grad(dist.log_prob(torch.from_numpy(action).float()), param)[0] * advantage

        for name, param in self.named_parameters():
            param.grad = grads[name]
            self.writer.add_scalar(f"grad_norm_{name}", torch.norm(param.grad), batch)
            print(name, param.grad)

        self.optimizer.step()
        return
