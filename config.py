class Parametrs:
    def __init__(self):
        self.state_dim = 30 #output dim from convolutions(fixed)
        self.action_dim = 3
        self.num_context = 5
        # learning_rates:
        self.lr_policy = 1e-3
        self.lr_value = 1e-3
        self.lr_descr = 1e-2

        #GAE parametrs
        self.gamma = 0.97
        self.lam = 0.95

        # actor loss parametrs
        self.entropy_coef = 1e-3
        self.adv_coef = 1

        # training parametrs
        self.max_episodes = 100000
        self.ep_len = 10
        self.eps_for_update = 5
        self.iters_value_train = 10

        # discriminator training parametrs
        self.eps_for_dc = 20  # collect trajectories for optimizing
        self.iters_discriminator_train = 10

        self.dc_hidden_dim = 32
        self.dc_num_layers = 1