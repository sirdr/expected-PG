import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import backward, Variable
import pdb

'''
Q-Critic is trained by SARSA (learning current policy's state-action values).
'''
class QCritic(nn.Module):
    def __init__(self, env, config, metrics_writer, use_gpu=False):
        super(QCritic, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]

        self.layers = [nn.Linear(self.state_space + self.action_space, config.qcritic_layers[0])]
        for i in range(1, len(config.qcritic_layers)):
            self.layers.append(nn.Linear(config.qcritic_layers[i-1], config.qcritic_layers[i]))
        self.layers.append(nn.Linear(config.qcritic_layers[-1], 1))
        self.layers = nn.ModuleList(self.layers)

        self.gamma = config.gamma

        self.optimizer = optim.Adam(self.parameters(), lr=config.critic_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.critic_lr_step_size, gamma=config.critic_lr_decay)

        self.metrics_writer = metrics_writer
        self.step = 0

    def forward(self, state, action):
        out = torch.cat((state, action), -1)
        for layer in self.layers[:-1]:
            out = layer(out)
            out = F.relu(out)
        return self.layers[-1](out)

    def apply_gradient(self, s1, a1, r, s2, a2, target_q=None):
        self.optimizer.zero_grad()
        current_Q = self.forward(torch.from_numpy(s1).float(), torch.from_numpy(a1).float())
        if target_q is None:
            if a2 is None:
                y = torch.tensor(r).float()
            else:
                y = r + self.gamma * self.forward(torch.from_numpy(s2).float(), torch.from_numpy(a2).float())
        else:
            if a2 is None:
                y = torch.tensor(r).float()
            else:
                y = r + self.gamma * target_q.forward(torch.from_numpy(s2).float(), torch.from_numpy(a2).float())
        loss = nn.MSELoss()(current_Q, y.detach())
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return

    def apply_gradient_expected(self, s1, a1, r, s2, policy, last_state=False, target_q=None):
        self.optimizer.zero_grad()
        current_Q = self.forward(torch.from_numpy(s1).float(), torch.from_numpy(a1).float())
        if not last_state:
            a2 = policy.get_actions(s2, 10)
            if target_q is None:
                y = r + self.gamma * torch.mean(self.forward(torch.from_numpy(s2).float().repeat(10,1), a2))
            else:
                y = r + self.gamma * torch.mean(target_q.forward(torch.from_numpy(s2).float().repeat(10,1), a2))
        else:
            y = torch.tensor(r).float()
        loss = nn.MSELoss()(current_Q, y.detach())
        loss.backward()
        self.optimizer.step()

        self.step += 1
        return

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards):
        self.optimizer.zero_grad()
        states = torch.tensor(np.array(ep_states[:-2])).float()
        actions = torch.tensor(np.array(ep_actions[:-1])).float()
        next_actions = torch.tensor(np.array(ep_actions[1:])).float()
        rewards = torch.tensor(ep_rewards[:-1]).float()
        next_states = torch.tensor(np.array(ep_states[1:-1])).float()
        current_Q = self.forward(states, actions).flatten()
        next_Q = rewards + self.gamma * self.forward(next_states, next_actions).flatten()
        loss = nn.MSELoss()(current_Q, next_Q.detach())
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
