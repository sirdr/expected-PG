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
    This variant uses backward() to train the policy in a batch fashion, using the standard REINFORCE update.
'''
class PolicyBackwardMC(nn.Module):
    def __init__(self, env, config, writer):
        super(PolicyBackwardMC, self).__init__()
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_space_high = torch.from_numpy(env.action_space.high).float()
        self.action_space_low = torch.from_numpy(env.action_space.low).float()

        self.l1 = nn.Linear(self.state_space, 16)
        self.l2 = nn.Linear(16, self.action_space)

        self.n_samples_per_state = config.n_samples_per_state

        if config.learn_std:
            self.log_std = torch.nn.Parameter(torch.tensor([np.log(0.2)], dtype=torch.float32), requires_grad=True)
        else:
            self.log_std = torch.tensor([np.log(0.2)], dtype=torch.float32)

        self.normalize_advantages = config.normalize_advantages

        self.gamma = config.gamma

        self.optimizer = optim.Adam(self.parameters(), lr=config.policy_lr)
        # self.optimizer = optim.SGD(self.parameters(), lr=config.policy_lr)
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

    def apply_gradient_batch(self, states, actions, rewards, batch, qcritic, vcritic):

        self.optimizer.zero_grad()

        n_episodes = len(states)
        n_states = sum([len(s) for s in states])
        states = torch.tensor(np.array([state for episode in states for state in episode[:-1]])).float()
        action_means = self.forward(states).repeat(self.n_samples_per_state, 1)

        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_means, std)
        sample = dist.sample()
        actions = torch.max(torch.min(sample, self.action_space_high), self.action_space_low)

        advantages = qcritic(states.repeat(self.n_samples_per_state, 1), actions).flatten().detach() - vcritic(states.repeat(self.n_samples_per_state,1)).flatten().detach()
        self.writer.add_scalar(f"average_advantage", torch.mean(advantages), batch)
        self.writer.add_scalar(f"std_advantage", torch.std(advantages), batch)
        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (.1 + torch.std(advantages))

        log_probs = torch.distributions.normal.Normal(action_means, std).log_prob(actions).flatten()

        loss = - torch.dot(log_probs, advantages) / self.n_samples_per_state / n_states
        loss.backward()

        for name, param in self.named_parameters():
            # print(name, param.grad)
            self.writer.add_scalar(f"grad_norm_{name}", torch.norm(param.grad), batch)

        torch.nn.utils.clip_grad_norm(self.parameters(), 1.) #Clip gradients for model stability.
        self.optimizer.step()

        return

    def apply_gradient(self, state, qcritic, vcritic):
        self.optimizer.zero_grad()

        states = torch.from_numpy(state).float()
        action_means = self.forward(states).repeat(self.n_samples_per_state, 1)

        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_means, std)
        sample = dist.sample()
        actions = torch.max(torch.min(sample, self.action_space_high), self.action_space_low)
        advantages = qcritic(states.repeat(self.n_samples_per_state, 1), actions).flatten().detach() - vcritic(states.repeat(self.n_samples_per_state, 1)).flatten().detach()

        if self.normalize_advantages:
            advantages = (advantages - torch.mean(advantages)) / (.1 + torch.std(advantages))

        log_probs = torch.distributions.normal.Normal(action_means, std).log_prob(actions).flatten()

        loss = -torch.dot(log_probs, advantages) / self.n_samples_per_state
        loss.backward()

        self.optimizer.step()
