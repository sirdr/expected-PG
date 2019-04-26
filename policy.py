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

class Policy(nn.Module):
    def __init__(self, env, config, writer):
        super(Policy, self).__init__()
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

        self.optimizer = optim.Adam(self.parameters(), lr=config.policy_lr)
        # self.optimizer = optim.SGD(self.parameters(), lr=config.policy_lr)
        self.writer = writer

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


class PolicyReinforce(Policy): # Previously: PolicyBackward
    def __init__(self, env, config, writer):
        super(PolicyReinforce, self).__init__(env, config, writer)
        self.normalize_advantages = config.normalize_advantages

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

'''
    Policy is trained using any policy gradient method.
    This variant uses backward() to train the policy in a batch fashion, using the standard REINFORCE update.
'''
class PolicyMC(Policy): # Previsouly: PolicyBackwardMC
    def __init__(self, env, config, writer):
        super(PolicyMC, self).__init__(env, config, writer)
        self.action_space_high = self.action_space_high.float()
        self.action_space_low = self.action_space_low.float()
        self.n_samples_per_state = config.n_samples_per_state
        self.normalize_advantages = config.normalize_advantages

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

        #torch.nn.utils.clip_grad_norm(self.parameters(), 1.) #Clip gradients for model stability.
        self.optimizer.step()

'''
    Policy is trained using expected policy gradients through numerical integration.
'''
class PolicyIntegration(Policy):
    def __init__(self, env, config, writer):
        super(PolicyIntegration, self).__init__(env, config, writer)
        # Episode policy and reward history
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def apply_gradient_batch(self, states, actions, rewards, batch, qcritic, vcritic=None):

        self.optimizer.zero_grad()
        grads = {}
        for name, param in self.named_parameters():
            grads[name] = torch.zeros_like(param)

        n_episodes = len(states)

        states = [state for episode in states for state in episode[:-1]]
        actions = [action for episode in actions for action in episode]
        n_actions = len(actions)

        std = torch.exp(self.log_std)

        for i in range(n_actions):

            state = states[i]
            action = actions[i]
            action_means = self.forward(torch.from_numpy(state).float())

            for name, param in self.named_parameters():
                tensor_state = torch.from_numpy(state).float()
                dist = torch.distributions.normal.Normal(self.forward(tensor_state), std)
                fun = lambda a : (grad(torch.exp(dist.log_prob(torch.from_numpy(a).float())), param, retain_graph=True)[0] * (qcritic(tensor_state, torch.from_numpy(a).float()) - vcritic(tensor_state))).detach().numpy()
                estimate = integration.compute_integral_asr(fun, self.action_space_low.numpy(), self.action_space_high.numpy(), 0.01)
                grads[name] -= torch.from_numpy(estimate).float()

        for name, param in self.named_parameters():
            param.grad = grads[name]
            self.writer.add_scalar(f"grad_norm_{name}", torch.norm(param.grad), batch)
            print(name, param.grad)

        self.optimizer.step()
        return

class PolicyIntegrationTrapezoidal(Policy):
    def __init__(self, env, config, writer, num_actions=1000):
        super(PolicyIntegrationTrapezoidal, self).__init__(env, config, writer)
        self.action_space_high = self.action_space_high.float()
        self.action_space_low = self.action_space_low.float()
        self.n_samples_per_state = config.n_samples_per_state
        self.normalize_advantages = config.normalize_advantages
        # Episode policy and reward history
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.num_actions = num_actions

    def apply_gradient_batch(self, states, actions, rewards, batch, qcritic, vcritic=None):

        self.optimizer.zero_grad()

        n_episodes = len(states)
        n_states = sum([len(s) for s in states])

        states =np.array([state for episode in states for state in episode[:-1]])  
        print(states.shape)

        num_states = states.shape[0]
        actions = np.linspace(self.action_space_low,self.action_space_high, num=self.num_actions)
        actions = np.squeeze(actions)
        print(actions.shape)

        weights = (actions[1:]-actions[:-1])
        states = np.reshape(np.tile(states, len(actions)), (len(actions)*num_states, -1))

        actions = np.tile(actions, (num_states))
        weights = np.tile(weights, (num_states))
        actions = actions[:, None]
        weights = weights[:, None]

        states = torch.tensor(states).float()# What's the point of this
        actions = torch.tensor(actions).float()
        weights = torch.tensor(weights).float()

        action_means = self.forward(states)

        print(states.shape)
        print(actions.shape)

        advantages = qcritic(states, actions).flatten().detach() - vcritic(states).flatten().detach()
        self.writer.add_scalar(f"average_advantage", torch.mean(advantages), batch)
        self.writer.add_scalar(f"std_advantage", torch.std(advantages), batch)
        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (.1 + torch.std(advantages))

        std = torch.exp(self.log_std)
        log_probs = torch.distributions.normal.Normal(action_means, std).log_prob(actions).flatten()
        probs = torch.exp(log_probs)

        integrand = probs*advantages
        integrand_reshaped = torch.reshape(integrand, [-1, self.num_actions])
        integrand_reshaped_avg = (integrand_reshaped[:, :-1] + integrand_reshaped[:, 1:])/2.0
        integrand_avg = torch.reshape(integrand_reshaped_avg, [-1, 1])
        weighted_integrand = weights*integrand_avg

        loss = -torch.sum(weighted_integrand)/num_states
        loss.backward()

        for name, param in self.named_parameters():
            # print(name, param.grad)
            self.writer.add_scalar(f"grad_norm_{name}", torch.norm(param.grad), batch)

        #torch.nn.utils.clip_grad_norm(self.parameters(), 1.) #Clip gradients for model stability.
        self.optimizer.step()
