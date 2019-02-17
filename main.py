import gym
import torch
from policy import Policy
from critic import Critic
from config import Config

env = gym.make('MountainCarContinuous-v0')
env.seed(1); torch.manual_seed(1);

config = Config()
policy = Policy(env, config)
critic = Critic(env, config)

observation = env.reset()
done = False

states = [observation]
actions = []
rewards = []

while not done:
    env.render()
    action = policy.get_action(observation)
    action_value = critic(torch.from_numpy(observation).float(), torch.from_numpy(action).float())
    print("Action value: ", action_value)
    observation, reward, done, info = env.step(action)
    states.append(observation)
    actions.append(action)
    rewards.append(reward)
    if(len(actions) >= 2):
        critic.apply_gradient(states[-2], actions[-2], rewards[-1], states[-1], actions[-1])

# Apply gradient for specified states. Uses critic to compute policy gradient.
policy.apply_gradient(states, critic)
