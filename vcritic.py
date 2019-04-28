import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import backward, Variable

'''
V-Critic is trained by TD (learning current policy value).
'''
class VCritic(nn.Module):
    def __init__(self, env, config):
        super(VCritic, self).__init__()
        self.state_space = env.observation_space.shape[0]

        # Input is a concatenation of state and action.
        self.l1 = nn.Linear(self.state_space, 10)
        # Output a single Q value for that state and action.
        self.l2 = nn.Linear(10, 1)

        self.gamma = config.gamma

        self.optimizer = optim.Adam(self.parameters(), lr=config.critic_lr)

    def forward(self, state):
        out = self.l1(state)
        out = F.relu(out)
        out = self.l2(out)
        return out

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

    def compute_returns(self, rewards_by_path):
        returns = []
        for reward_path in rewards_by_path:
            g = np.array(reward_path)
            n_transitions = len(g)
            g = (self.gamma ** np.arange(n_transitions)) * g
            g = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
            returns.append(g)
        returns = np.concatenate(returns)
        return returns
