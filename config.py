class InvertedPendulumConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 5e-4
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = 0.99
    vcritic_layers = [10]
    qcritic_layers = [48]
    policy_layers = [16]

class CheetahConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 5e-4
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [10]
    qcritic_layers = [48]
    policy_layers = [16]

class WalkerConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 5e-4
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [10]
    qcritic_layers = [48]
    policy_layers = [16]

class ReacherConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 5e-4
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [10]
    qcritic_layers = [48]
    policy_layers = [16]
