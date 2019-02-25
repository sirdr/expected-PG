import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
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

        self.l1 = nn.Linear(self.state_space, 24)
        self.l2 = nn.Linear(24, self.action_space)

        self.gamma = config.gamma
        self.sigma = config.sigma
        self.eps = 1.0

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
        # out = self.action_space_low.float() + (self.action_space_high.float() - self.action_space_low.float()) * F.sigmoid(out)
        return out

    def get_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_mean = self.forward(Variable(state))
        m = torch.distributions.normal.Normal(action_mean, self.sigma)
        sample = m.sample().numpy()
        print("Action mean: {}".format(action_mean))
        return np.clip(sample, self.action_space_low.numpy(), self.action_space_high.numpy())

    def apply_gradient2(self, state, qcritic, vcritic, step):
        self.optimizer.zero_grad()
        for param in self.parameters():
            param_grad = np.zeros(param.shape)
            prob = lambda a : 1./(2*np.pi*self.sigma**2)**(self.action_space/2.) * torch.exp(-torch.norm((torch.from_numpy(a).float() - self.forward(torch.from_numpy(state).float())))**2 / (2*self.sigma**2))
            param_grad += (self.gamma**step) * integration.compute_integral_asr(f, self.action_space_low.numpy(), self.action_space_high.numpy(), 0.1)
            param.grad = torch.from_numpy(-param_grad).float()
        self.optimizer.step()
        return

    def apply_gradient3(self, state, action, qcritic, vcritic, step):
        self.optimizer.zero_grad()
        for param in self.parameters():
            param_grad = np.zeros(param.shape)
            prob = lambda a : 1./(2*np.pi*self.sigma**2)**(self.action_space/2.) * torch.exp(-torch.norm((torch.from_numpy(a).float() - self.forward(torch.from_numpy(state).float())))**2 / (2*self.sigma**2))
            param_grad = (self.gamma ** step) * (prob(action) * torch.autograd.grad(torch.log(prob(action) + .0001), param)[0] * (qcritic(torch.from_numpy(state).float(), torch.from_numpy(action).float()) - vcritic(torch.from_numpy(state).float())))
            param.grad = -param_grad
            # print(param.grad)
        self.optimizer.step()
        return

    '''
    Without V critic
    '''
    def apply_gradient4(self, states, actions, rewards, qcritic, vcritic, ep):
        self.eps *= 0.95
        self.optimizer.zero_grad()

        n_states = len(states)

        rewards = np.array(rewards)
        g = self.gamma ** np.arange(n_states) * rewards
        g = np.cumsum(g[::-1])[::-1]

        for param in self.parameters():
            param.grad = torch.zeros(param.shape)

        for i in range(n_states):
            state = states[i]
            action = actions[i]
            for param in self.parameters():
                prob = lambda a : 1./(2*np.pi*self.sigma**2)**(self.action_space/2.) * torch.exp(-torch.norm((torch.from_numpy(a).float() - self.forward(torch.from_numpy(state).float())))**2 / (2*self.sigma**2))
                grad = - 1./(self.sigma**2) * torch.autograd.grad(self.forward(torch.from_numpy(state).float()), param)[0] * (torch.from_numpy(action).float() - self.forward(torch.from_numpy(state).float()))
                param.grad -= grad * g[i]
        for name, param in self.named_parameters():
            self.writer.add_scalar("gradient_norm_{}".format(name), torch.norm(param.grad), ep)
        self.optimizer.step()
        return

    '''
    Without V critic
    '''
    def apply_gradient5(self, states, actions, rewards, qcritic, vcritic):
        self.optimizer.zero_grad()

        n_states = len(states)

        rewards = np.array(rewards)
        g = self.gamma ** np.arange(n_states) * rewards
        g = np.cumsum(g[::-1])[::-1]

        for param in self.parameters():
            param.grad = torch.zeros(param.shape)

        for i in range(n_states):
            state = states[i]
            action = actions[i]
            for param in self.parameters():
                prob = lambda a : 1./(2*np.pi*self.sigma**2)**(self.action_space/2.) * torch.exp(-torch.norm((torch.from_numpy(a).float() - self.forward(torch.from_numpy(state).float())))**2 / (2*self.sigma**2))
                grad = - 1./(self.sigma**2) * torch.autograd.grad(self.forward(torch.from_numpy(state).float()), param)[0] * (torch.from_numpy(action).float() - self.forward(torch.from_numpy(state).float()))
                param.grad += grad * g[i]
            # print(param.grad)
        self.optimizer.step()
        return
