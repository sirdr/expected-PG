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
class PolicyBackward(nn.Module):
    def __init__(self, env, config, writer):
        super(PolicyBackward, self).__init__()
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_space_high = torch.from_numpy(env.action_space.high)
        self.action_space_low = torch.from_numpy(env.action_space.low)

        self.l1 = nn.Linear(self.state_space, 16)
        self.l2 = nn.Linear(16, self.action_space)

        if config.learn_std:
            self.log_std = torch.nn.Parameter(torch.tensor([np.log(0.2)], dtype=torch.float32), requires_grad=True)
        else:
            self.log_std = torch.tensor([np.log(0.2)], dtype=torch.float32)

        self.gamma = config.gamma

        self.normalize_advantages = config.normalize_advantages
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

    def compute_advantages(self, states, rewards_by_path, vcritic=None, normalize=True):
        advantages = []
        for i in range(len(rewards_by_path)):
            g = np.array(rewards_by_path[i])
            n_transitions = len(g)
            g = (self.gamma ** np.arange(n_transitions)) * g
            g = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
            advantages.append(g)
        advantages = np.concatenate(advantages)
        if vcritic != None:
            v_values = vcritic(torch.from_numpy(np.vstack(states)).float()).detach().numpy().flatten()
            advantages -= v_values
        if normalize:
            advantages = (advantages - np.mean(advantages)) / (.1 + np.std(advantages))
        return advantages

    def apply_gradient_batch(self, states, actions, rewards, batch, vcritic):

        self.optimizer.zero_grad()

        n_episodes = len(states)
        n_states = sum([len(s) for s in states])
        states = torch.tensor(np.array([state for episode in states for state in episode[:-1]])).float()
        actions = torch.tensor(np.array([action for episode in actions for action in episode])).float()
        advantages = torch.tensor(self.compute_advantages(states, rewards, vcritic=vcritic, normalize=self.normalize_advantages)).float()

        self.writer.add_scalar(f"average_advantage", torch.mean(advantages), batch)
        self.writer.add_scalar(f"std_advantage", torch.std(advantages), batch)

        std = torch.exp(self.log_std)
        log_probs = torch.distributions.normal.Normal(self.forward(states), std).log_prob(actions).flatten()

        loss = - torch.dot(log_probs, advantages) / n_states
        loss.backward()

        for name, param in self.named_parameters():
            # print(name, param.grad)
            self.writer.add_scalar(f"grad_norm_{name}", torch.norm(param.grad), batch)

        # torch.nn.utils.clip_grad_norm(self.parameters(), 1.) #Clip gradients for model stability.
        self.optimizer.step()

        return
