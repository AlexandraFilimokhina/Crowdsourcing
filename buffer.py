import torch
import numpy as np
import scipy.signal
from collections import namedtuple, deque

class Rollout:
    def __init__(self, ep_len, eps_for_dc, eps_for_update, gamma, lam):
        self.gamma, self.lam = gamma, lam  # for GAE computation

        self.eps_for_dc = eps_for_dc

        self.ep_len = ep_len
        self.buffer_size = eps_for_update * ep_len
        self.dc_buffer_size = eps_for_dc * ep_len

        self.transition = namedtuple('Transition', ('states_pictures', 'context_one_hot', 'actions_one_hot',
                                                    'rewards', 'values'))
        self.dc_transition = namedtuple('dc_Transition', ('states_pictures', 'actions_one_hot'))
        self.reset_buff()
        self.reset_dc_buff()

    def reset_buff(self):
        self.memory = deque(maxlen=self.buffer_size)
        self.advantages = deque(maxlen=self.buffer_size)
        self.disc_rewards = deque(maxlen=self.buffer_size)
        self.logs_context = deque(maxlen=self.buffer_size)

    def reset_dc_buff(self):
        self.dc_memory = deque(maxlen=self.dc_buffer_size)
        self.dc_context = deque(maxlen=self.eps_for_dc)


    def push(self, state, context_one_hot, action, reward, value):
        self.memory.append(self.transition(state, context_one_hot, action, reward, value))
        self.dc_memory.append(self.dc_transition(state, self.make_one_hot_act(action)))


    def postprocessing(self, log_p_c, context):

        batch = self.transition(*zip(*self.memory))
        rewards = np.append(batch.rewards[-self.ep_len:], np.array([0]))
        values = np.append(batch.values[-self.ep_len:], np.array([0]))
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.disc_rewards.extend(self.discount_cumsum(rewards, self.gamma)[:-1])
        self.advantages.extend(self.discount_cumsum(deltas, self.gamma * self.lam))
        self.logs_context.extend([float(log_p_c)]*self.ep_len)

        self.dc_context.append(context)

    def get_buffer(self):
        batch = self.transition(*zip(*self.memory))
        return torch.cat(batch.states_pictures), \
               torch.cat(batch.context_one_hot),\
               torch.FloatTensor(batch.actions_one_hot),\
               torch.FloatTensor(self.normalize(self.advantages)),\
               torch.FloatTensor(self.normalize(self.logs_context)),\
               torch.FloatTensor(self.disc_rewards)

    def get_dc_buff(self, context=None):
        batch = self.dc_transition(*zip(*self.dc_memory))
        if context is not None:
            return torch.cat(batch.states_pictures[-self.ep_len:]),\
                   torch.FloatTensor(batch.actions_one_hot[-self.ep_len:]), \
                   torch.FloatTensor(context)
        return torch.cat(batch.states_pictures),\
               torch.FloatTensor(batch.actions_one_hot), \
               torch.FloatTensor(self.dc_context)


    @staticmethod
    def make_one_hot_act(action):
        action_one_not = np.zeros(3)
        action_one_not[action] = 1
        return action_one_not

    @staticmethod
    def discount_cumsum(x, coef):
        return scipy.signal.lfilter([1], [1, float(-coef)], x[::-1], axis=0)[::-1]

    @staticmethod
    def normalize(array):
        return (array - np.mean(array)) / (np.std(array) + 1e-5)