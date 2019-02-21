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
        self.l1 = nn.Linear(self.state_space + self.action_space, 10)
        # Output a single Q value for that state and action.
        self.l2 = nn.Linear(10, 1)

        self.gamma = config.gamma

        self.optimizer = optim.Adam(self.parameters(), lr=config.critic_lr)

    def forward(self, state, action):
        concat = torch.cat((state, action), 0)
        out = self.l1(concat)
        out = F.relu(out)
        out = self.l2(out)
        return out

    def apply_gradient(self, s1, a1, r, s2, a2):
        self.optimizer.zero_grad()
        current_Q = self.forward(torch.from_numpy(s1).float(), torch.from_numpy(a1).float())
        # delta = r + self.gamma * self.forward(torch.from_numpy(s2).float(), torch.from_numpy(a2).float()) - current_Q
        loss = nn.MSELoss()(current_Q, r + self.gamma * self.forward(torch.from_numpy(s2).float(), torch.from_numpy(a2).float()))
        loss.backward()
        self.optimizer.step()
        return
