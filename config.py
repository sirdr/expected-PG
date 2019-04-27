class Config:
    gamma = 1.00
    eps = 0.01
    critic_lr = 1e-2
    policy_lr = 1e-3
    n_samples_per_state = 100
    normalize_advantages = True
    learn_std = False
    tau = 0.9
