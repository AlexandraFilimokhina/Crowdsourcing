class Parametrs:
    def __init__(self):
        self.state_dim = 30
        self.num_context = 5
        # learning_rates:
        self.lr_policy = 1e-3
        self.lr_value = 1e-3
        self.lr_descr = 5e-2

        # actor loss parametrs
        self.entropy_coef = 1e-3
        self.adv_coef = 1e-3

        # training parametrs
        self.ep_len = 10
        self.iters_value_train = 10
        self.iters_discriminator_train = 5
        self.num_trajectories = 100000
        self.train_dc_interv = 5 #collect trajectories for optimizing
