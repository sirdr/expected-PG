class InvertedPendulumConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-3
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = 0.99
    vcritic_layers = [16]
    qcritic_layers = [48]
    policy_layers = [16]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1

class CheetahConfig:
    gamma = 0.9
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-4
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [48]
    qcritic_layers = [48,32]
    policy_layers = [32,32,32]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1

class WalkerConfig:
    gamma = 0.9
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-4
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [32]
    qcritic_layers = [48]
    policy_layers = [16,16]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1

class ReacherConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-3
    policy_lr = 1e-4
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [32]
    qcritic_layers = [48]
    policy_layers = [32]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1

class LanderConfig:
    gamma = 0.99
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-3
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [32]
    qcritic_layers = [48]
    policy_layers = [32]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1
