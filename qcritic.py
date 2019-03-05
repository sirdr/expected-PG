import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import backward, Variable

'''
Q-Critic is trained by SARSA (learning current policy's state-action values).
'''
class QCritic(nn.Module):
    def __init__(self, env, config):
        super(QCritic, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        # Upper and lower bound on action space (it is a box).
        self.action_space_high = env.action_space.high
        self.action_space_low = env.action_space.low

        # Input is a concatenation of state and action.
        self.l1 = nn.Linear(self.state_space + self.action_space, 24)
        # Output a single Q value for that state and action.
        self.l2 = nn.Linear(24, 48)
        self.l3 = nn.Linear(48, 1)

        self.gamma = config.gamma

        self.optimizer = optim.Adam(self.parameters(), lr=config.critic_lr)

    def forward(self, state, action):
        concat = torch.cat((state, action), -1)
        out = self.l1(concat)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        return out

    def apply_gradient(self, s1, a1, r, s2, a2):
        self.optimizer.zero_grad()
        current_Q = self.forward(torch.from_numpy(s1).float(), torch.from_numpy(a1).float())
        # print(f"Current Q: {current_Q} - Target: {r + self.gamma * self.forward(torch.from_numpy(s2).float(), torch.from_numpy(a2).float())}")
        loss = nn.MSELoss()(current_Q, r + self.gamma * self.forward(torch.from_numpy(s2).float(), torch.from_numpy(a2).float()))
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

    def apply_gradient_batch(self, states, actions, rewards):
        self.optimizer.zero_grad()
        returns = self.compute_returns(rewards)
        states = np.vstack([state for episode in states for state in episode[:-1]])
        actions = np.vstack([action for episode in actions for action in episode])
        current_Q = self.forward(torch.from_numpy(states).float(), torch.from_numpy(actions).float())
        loss = nn.MSELoss()(current_Q, torch.from_numpy(returns).float())
        loss.backward()
        self.optimizer.step()
        return
