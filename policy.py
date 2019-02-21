import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import integration

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

        self.l1 = nn.Linear(self.state_space, 10)
        self.l2 = nn.Linear(10, self.action_space)

        self.gamma = config.gamma
        self.sigma = config.sigma

        self.optimizer = optim.Adam(self.parameters(), lr=config.policy_lr)

        # Episode policy and reward history
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    '''
        Estimate the mean of a Gaussian (continuous) stochastic policy.
    '''
    def forward(self, state):
        out = self.l1(state)
        out = F.relu(out)
        out = self.l2(out)
        # out = self.action_space_low + (self.action_space_high - self.action_space_low) * F.sigmoid(out)
        return out

    def get_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_mean = self.forward(Variable(state))
        m = torch.distributions.normal.Normal(action_mean, self.sigma)
        # print(action_mean)
        sample = m.sample().numpy()
        return np.clip(sample, self.action_space_low.numpy(), self.action_space_high.numpy())

    # def apply_gradient(self, states, critic, ep):
    #     self.optimizer.zero_grad()
    #     n_states = len(states)
    #     for param in self.parameters():
    #         param_grad = np.zeros(param.shape)
    #         for state in states:
    #             prob = lambda a : 1./(2*np.pi*self.sigma**2)**(self.action_space/2.) * torch.exp(-torch.norm((torch.from_numpy(a).float() - self.forward(torch.from_numpy(state).float())))**2 / (2*self.sigma**2))
    #             f = lambda a : (torch.autograd.grad(prob(a), param)[0] * critic(torch.from_numpy(state).float(), torch.from_numpy(a).float())).detach().numpy()
    #             param_grad += 1./n_states * integration.compute_integral(f, self.action_space_low, self.action_space_high, param.shape, 0.1)
    #         param.grad = torch.from_numpy(-param_grad).float()
    #         print(param.grad)
    #     self.optimizer.step()
    #     return

    def apply_gradient2(self, state, qcritic, vcritic, step):
        self.optimizer.zero_grad()
        for param in self.parameters():
            param_grad = np.zeros(param.shape)
            prob = lambda a : 1./(2*np.pi*self.sigma**2)**(self.action_space/2.) * torch.exp(-torch.norm((torch.from_numpy(a).float() - self.forward(torch.from_numpy(state).float())))**2 / (2*self.sigma**2))
            # f = lambda a : (prob(a) * torch.autograd.grad(torch.log(prob(a) + .0001), param)[0] * (qcritic(torch.from_numpy(state).float(), torch.from_numpy(a).float()) - vcritic(torch.from_numpy(state).float()))).detach().numpy()
            f = lambda a : (torch.autograd.grad(prob(a), param)[0] * qcritic(torch.from_numpy(state).float(), torch.from_numpy(a).float()) - vcritic(torch.from_numpy(state).float())).detach().numpy()
            param_grad += (self.gamma**step) * integration.compute_integral(f, self.action_space_low, self.action_space_high, param.shape, 0.1)
            param.grad = torch.from_numpy(-param_grad).float()
            # print(param.grad)
        self.optimizer.step()
        return
