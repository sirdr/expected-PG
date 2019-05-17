import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import backward, Variable
import pdb

'''
V-Critic is trained by TD (learning current policy value).
'''
class VCritic(nn.Module):
    def __init__(self, env, config):
        super(VCritic, self).__init__()
        self.state_space = env.observation_space.shape[0]

        self.layers = [nn.Linear(self.state_space, config.vcritic_layers[0])]
        for i in range(1, len(config.vcritic_layers)):
            self.layers.append(nn.Linear(config.vcritic_layers[i-1], config.vcritic_layers[i]))
        self.layers.append(nn.Linear(config.vcritic_layers[-1], 1))
        self.layers = nn.ModuleList(self.layers)

        self.gamma = config.gamma
        self.optimizer = optim.Adam(self.parameters(), lr=config.critic_lr)

    def forward(self, out):
        for layer in self.layers[:-1]:
            out = layer(out)
            out = F.relu(out)
        return self.layers[-1](out)

    def apply_gradient(self, s1, a1, r, s2):
        self.optimizer.zero_grad()
        current_V = self.forward(torch.from_numpy(s1).float())
        if s2 is None:
            next_V = torch.tensor(r).float()
        else:
            next_V = r + self.gamma * self.forward(torch.from_numpy(s2).float())
        loss = nn.MSELoss()(current_V, next_V.detach())
        loss.backward()
        self.optimizer.step()
        return

    def apply_gradient_episode(self, ep_states, ep_rewards):
        self.optimizer.zero_grad()
        current_V = self.forward(torch.tensor(np.array(ep_states[:-1])).float()).flatten()
        true_V = torch.tensor(self.compute_returns(ep_rewards)).float()
        loss = nn.MSELoss()(current_V, true_V.detach())
        loss.backward()
        self.optimizer.step()
        return

    def compute_returns(self, ep_rewards):
        g = np.array(ep_rewards)
        n_transitions = len(g)
        g = (self.gamma ** np.arange(n_transitions)) * g
        g = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
        return g

    def compute_returns_batch(self, rewards_by_path):
        returns = []
        for reward_path in rewards_by_path:
            g = np.array(reward_path)
            n_transitions = len(g)
            g = (self.gamma ** np.arange(n_transitions)) * g
            g = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
            returns.append(g)
        returns = np.concatenate(returns)
        return returns
