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
    def __init__(self, env, config, metrics_writer):
        super(Policy, self).__init__()
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_space_high = torch.from_numpy(env.action_space.high).float()
        self.action_space_low = torch.from_numpy(env.action_space.low).float()

        self.l1 = nn.Linear(self.state_space, 16)
        self.l2 = nn.Linear(16, self.action_space)

        if config.learn_std:
            self.log_std = torch.nn.Parameter(torch.tensor([np.log(0.2)], dtype=torch.float32), requires_grad=True)
        else:
            self.log_std = torch.tensor([np.log(0.2)], dtype=torch.float32)

        self.gamma = config.gamma
        self.normalize_advantages = config.normalize_advantages

        self.optimizer = optim.Adam(self.parameters(), lr=config.policy_lr)
        self.metrics_writer = metrics_writer

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
    def __init__(self, env, config, metrics_writer):
        super(PolicyReinforce, self).__init__(env, config, metrics_writer)
        self.normalize_advantages = config.normalize_advantages

    def compute_advantages(self, states, rewards, vcritic=None, normalize=True):
        g = np.array(rewards)
        n_transitions = len(g)
        g = (self.gamma ** np.arange(n_transitions)) * g
        advantages = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
        if vcritic != None:
            v_values = vcritic(torch.from_numpy(np.vstack(states)).float()).detach().numpy().flatten()
            advantages -= v_values
        if normalize:
            advantages = (advantages - np.mean(advantages)) / (.1 + np.std(advantages))
        return advantages

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, vcritic):

        self.optimizer.zero_grad()
        n_states = len(ep_states)
        states = torch.tensor(np.array([state for state in ep_states[:-1]])).float()
        actions = torch.tensor(np.array([action for action in ep_actions])).float()
        advantages = torch.tensor(self.compute_advantages(states, ep_rewards, vcritic=vcritic, normalize=self.normalize_advantages)).float()

        std = torch.exp(self.log_std)
        log_probs = torch.distributions.normal.Normal(self.forward(states), std).log_prob(actions).flatten()

        loss = - torch.dot(log_probs, advantages) / n_states
        loss.backward()

        self.metrics_writer.write_metric(episode, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(episode, "std_advantage", torch.std(advantages))
        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))

        self.optimizer.step()

        return

'''
    Policy is trained using any policy gradient method.
    This variant uses backward() to train the policy in an episodic fashion, using the standard REINFORCE update.
'''
class PolicyMC(Policy):
    def __init__(self, env, config, metrics_writer):
        super(PolicyMC, self).__init__(env, config, metrics_writer)
        self.n_samples_per_state = config.n_samples_per_state

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic):

        self.optimizer.zero_grad()

        n_states = len(ep_states)
        states = torch.tensor(np.array([state for state in ep_states[:-1]])).float()
        action_means = self.forward(states).repeat(self.n_samples_per_state, 1)

        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_means, std)
        sample = dist.sample()
        actions = torch.max(torch.min(sample, self.action_space_high), self.action_space_low)

        advantages = qcritic(states.repeat(self.n_samples_per_state, 1), actions).flatten().detach() - vcritic(states.repeat(self.n_samples_per_state,1)).flatten().detach()
        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (0.1 + torch.std(advantages))

        log_probs = torch.distributions.normal.Normal(action_means, std).log_prob(actions).flatten()

        loss = - torch.dot(log_probs, advantages) / self.n_samples_per_state / n_states
        loss.backward()

        self.metrics_writer.write_metric(episode, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(episode, "std_advantage", torch.std(advantages))
        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))

        self.optimizer.step()

'''
    Policy is trained using expected policy gradients through numerical integration.
'''
class PolicyIntegration(Policy):
    def __init__(self, env, config, metrics_writer):
        super(PolicyIntegration, self).__init__(env, config, metrics_writer)

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic=None):

        self.optimizer.zero_grad()
        grads = {}
        for name, param in self.named_parameters():
            grads[name] = torch.zeros_like(param)

        states = [state for state in ep_states[:-1]]
        n_actions = len(ep_actions)

        std = torch.exp(self.log_std)

        # Manually compute gradients
        for i in range(n_actions):

            state = states[i]
            action_means = self.forward(torch.from_numpy(state).float())

            for name, param in self.named_parameters():
                tensor_state = torch.from_numpy(state).float()
                dist = torch.distributions.normal.Normal(self.forward(tensor_state), std)
                fun = lambda a : (grad(torch.exp(dist.log_prob(torch.from_numpy(a).float())), param, retain_graph=True)[0] * (qcritic(tensor_state, torch.from_numpy(a).float()) - vcritic(tensor_state))).detach().numpy()
                estimate = integration.compute_integral_asr(fun, self.action_space_low.numpy(), self.action_space_high.numpy(), 0.01)
                grads[name] -= torch.from_numpy(estimate).float()

        # Manually assign gradients to the right tensors.
        for name, param in self.named_parameters():
            param.grad = grads[name]
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))
            print(name, param.grad)

        # Take an optimizer step.
        self.optimizer.step()
        return


####### TODO: STILL NEED TO MODIFY THAT ONE!

class PolicyIntegrationTrapezoidal(Policy):
    def __init__(self, env, config, metrics_writer, num_actions=1000):
        super(PolicyIntegrationTrapezoidal, self).__init__(env, config, metrics_writer)
        self.n_samples_per_state = config.n_samples_per_state
        self.num_actions = num_actions

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic=None):

        self.optimizer.zero_grad()

        states = np.array([state for state in ep_states[:-1]])
        num_states = states.shape[0]

        # TODO: Might not work in higher dimensions (>1)! Check linspace.
        actions = np.linspace(self.action_space_low, self.action_space_high, num=self.num_actions)
        actions = np.squeeze(actions)
        weights = (actions[1:]-actions[:-1])

        states = np.reshape(np.tile(states, len(actions)), (len(actions)*num_states, -1))

        actions = np.tile(actions, (num_states))
        weights = np.tile(weights, (num_states))
        actions = actions[:, None]
        weights = weights[:, None]

        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        weights = torch.tensor(weights).float()

        # states = torch.tensor(np.reshape(np.tile(states, len(actions)), (len(actions) * num_states, -1))).float()

        # actions = torch.tensor(actions).float().repeat(num_states, 1)
        # weights = torch.tensor(weights).float().repeat(num_states, 1)

        action_means = self.forward(states)

        advantages = qcritic(states, actions).flatten().detach() - vcritic(states).flatten().detach()

        self.metrics_writer.write_metric(episode, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(episode, "std_advantage", torch.std(advantages))
        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (.1 + torch.std(advantages))

        std = torch.exp(self.log_std)
        log_probs = torch.distributions.normal.Normal(action_means, std).log_prob(actions).flatten()
        probs = torch.exp(log_probs)

        integrand = probs * advantages
        integrand_reshaped = torch.reshape(integrand, [-1, self.num_actions])
        integrand_reshaped_avg = (integrand_reshaped[:, :-1] + integrand_reshaped[:, 1:])/2.0
        integrand_avg = torch.reshape(integrand_reshaped_avg, [-1, 1])
        weighted_integrand = weights*integrand_avg

        loss = -torch.sum(weighted_integrand) / num_states
        loss.backward()

        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))

        self.optimizer.step()
