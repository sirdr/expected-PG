class InvertedPendulumConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    # critic_lr = 1e-3
    policy_lr = 1e-3
    # policy_lr = 3e-4
    policy_lr_decay = 0.95
    critic_lr_decay = 0.95
    policy_lr_step_size = 200
    critic_lr_step_size = 200000
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = 0.99
<<<<<<< HEAD
    vcritic_layers = [16]
    qcritic_layers = [48]
    policy_layers = [16]
    # vcritic_layers = [32,32]
    # qcritic_layers = [32,32]
    # policy_layers = [32,32]
=======
    vcritic_layers = [32,16]
    qcritic_layers = [48,16]
    policy_layers = [32,32]
>>>>>>> ba30c1cc30fdfab77e28a968aa30235436384044
    action_std = 0.2
    clip_actions=True
    clip_grad = 1
    num_episodes = 4000
    clever=False

class CheetahConfig:
    gamma = 0.9
    eps = 0.01
    # critic_lr = 1e-3
    critic_lr = 1e-3
    policy_lr = 1e-4
    # policy_lr = 5e-5
    policy_lr_decay = 0.95
    critic_lr_decay = 0.95
    policy_lr_step_size = 200
    critic_lr_step_size = 200000
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [32,32]
    qcritic_layers = [32,32]
    policy_layers = [32,32]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1
    num_episodes = 5000
    clever=False

class WalkerConfig:
    gamma = 0.9
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-4
    policy_lr_decay = 0.95
    critic_lr_decay = 0.95
    policy_lr_step_size = 200
    critic_lr_step_size = 200000
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = True
    tau = .99
    vcritic_layers = [32,16]
    qcritic_layers = [48,16]
    policy_layers = [32,32]
    action_std = 0.2
    clip_actions=True
    clip_grad = 2
    num_episodes = 5000

class HopperConfig:
    gamma = .99
    eps = 0.01
    critic_lr = 1e-3
    policy_lr = 1e-5
    policy_lr_decay = 0.95
    critic_lr_decay = 0.95
    policy_lr_step_size = 200
    critic_lr_step_size = 200000
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    vcritic_layers = [32,32]
    qcritic_layers = [32,32]
    policy_layers = [32,32]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1
    num_episodes = 5000
    clever=False

class ReacherConfig:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-3
    policy_lr_decay = 0.95
    critic_lr_decay = 0.95
    policy_lr_step_size = 200
    critic_lr_step_size = 200000
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = .99
    # vcritic_layers = [32,16]
    # qcritic_layers = [48,16]
    # policy_layers = [32,32]
    vcritic_layers = [32,32]
    qcritic_layers = [32,32]
    policy_layers = [32,32]
    action_std = 0.2
    clip_actions=True
    clip_grad = 1
    num_episodes = 5000
    clever=False

class LanderConfig:
    gamma = 0.99
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-3
    policy_lr_decay = 0.95
    critic_lr_decay = 0.95
    policy_lr_step_size = 200
    critic_lr_step_size = 200000
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
    num_episodes = 5000
    clever=False
