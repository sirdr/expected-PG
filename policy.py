import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

'''
    Policy is trained using any policy gradient method.
'''
class Policy(nn.Module):
    def __init__(self, env, config):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_space_high = torch.from_numpy(env.action_space.high)
        self.action_space_low = torch.from_numpy(env.action_space.low)

        self.l1 = nn.Linear(self.state_space, 40)
        self.l2 = nn.Linear(40, self.action_space)

        self.gamma = config.gamma
        self.sigma = config.sigma

        # Episode policy and reward history
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, state):
        out = self.l1(state)
        out = F.relu(out)
        out = self.l2(out)
        out = self.action_space_low + (self.action_space_high - self.action_space_low) * F.sigmoid(out)
        return out

    def get_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_mean = self.forward(Variable(state))
        m = torch.distributions.normal.Normal(action_mean, self.sigma)
        sample = m.sample().numpy()
        print(sample)
        return np.clip(sample, self.action_space_low.numpy(), self.action_space_high.numpy())

    def apply_gradient(states, critic):
        return
