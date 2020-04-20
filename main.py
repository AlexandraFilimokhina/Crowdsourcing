import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from env import CroudsorsingEnv
from buffer import Rollout
from config import Parametrs
from models import Actor, Critic, Discriminator


class VALOR:
    def __init__(self):

        self.args = Parametrs()

        self.actor = Actor(state_dim=self.args.state_dim, context_dim=self.args.num_context)
        self.critic = Critic(state_dim=self.args.state_dim, context_dim=self.args.num_context)
        self.discriminator = Discriminator(input_dim=self.args.state_dim+3, context_dim=self.args.num_context)#+3 for action one hot

        self.buffer = Rollout(context_dim=self.args.num_context, state_dim=30, ep_len=self.args.ep_len, dc_interv=self.args.train_dc_interv)

        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.lr_policy)
        self.value_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.lr_value)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.args.lr_descr)

        self.context = Categorical(logits=torch.Tensor(np.ones(self.args.num_context)))

    def act(self, state, context_one_hot):
        with torch.no_grad():
            logits, x_concat = self.actor(state, context_one_hot=context_one_hot)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return action.numpy(), x_concat.squeeze().numpy()

    def eval_actions(self, states, actions):
        logits, _ = self.actor(states)

        policy = Categorical(logits=logits)
        log_prob = policy.log_prob(actions).unsqueeze(1)
        return log_prob

    def update(self, train_descriminator=False):

        self.state, self.action, self.advantage, self.q_z, self.reward = [torch.Tensor(x) for x in
                                                                          self.buffer.get_buffer()]
        self.__optimize_policy()
        self.__optimize_value()

        if train_descriminator:
            self.__optimize_discriminator()

    def __optimize_policy(self):
        self.actor.train()
        log_prob = self.eval_actions(self.state, self.action)
        entropy = (-log_prob * torch.exp(log_prob)).mean()
        L_policy = -(log_prob * (self.args.adv_coef * self.advantage + self.q_z)).mean() - self.args.entropy_coef * entropy

        self.policy_optimizer.zero_grad()
        L_policy.backward()
        self.policy_optimizer.step()

        self.actor.eval()

    def __optimize_value(self):
        self.critic.train()
        for _ in range(self.args.iters_value_train):
            value = self.critic(self.state)
            L_value = F.mse_loss(value.squeeze(), self.reward)

            self.value_optimizer.zero_grad()
            L_value.backward()
            self.value_optimizer.step()
        self.critic.eval()

    def __optimize_discriminator(self):
        self.discriminator.train()
        state, action, context = self.preprocess_dc_input(*self.buffer.get_dc_buff())
        self.buffer.reset_dc_buff()

        for _ in range(self.args.iters_discriminator_train):
            _, log_q, _ = self.discriminator(state, action, context)
            L_discriminator = - log_q.mean()

            self.discriminator_optimizer.zero_grad()
            L_discriminator.backward()
            self.discriminator_optimizer.step()

        print("-" * 150)
        print(f"L_discriminator :{L_discriminator:.3f}")
        self.discriminator.eval()

    @staticmethod
    def preprocess_dc_input(state, action, context):
        return torch.cat(state), torch.Tensor(action).float(), torch.Tensor(context)

    def train(self):
        self.critic.eval()
        episodes = 0
        rewards = []
        skill_rewards = []

        for t in range(self.args.num_trajectories):
            actions = []
            state = env.reset()
            reward, total_reward, done = 0, 0, False
            context = self.context.sample()

            context_one_hot = F.one_hot(context, self.args.num_context).squeeze().float()
            while not done:
                action, state_context = self.act(state, context_one_hot=context_one_hot.reshape(1, -1))
                value = self.critic(state, context_one_hot=context_one_hot.reshape(1, -1))

                next_state, reward, done, _ = env.step(action.item())

                self.buffer.push(state, state_context, action, reward, value.item())
                actions.append(action[0])
                total_reward += reward
                state = next_state

            dc_state = self.buffer.dc_state[-self.args.ep_len:]
            dc_action = self.buffer.dc_one_hot_act[-self.args.ep_len:]

            label, log_q, log_p = self.discriminator(*self.preprocess_dc_input(dc_state, dc_action, [float(context)]))

            self.buffer.postprocessing_ep(log_q.detach().numpy(), context)
            skill_rewards.append(log_q.item())
            rewards.append(total_reward )
            print("-"*150)
            print(f"Ep. {episodes} ||| context: {context.item()} ||| label: {label.item()}"
                  f" ||| skill_r: {log_q:.3f} ||| mean_s_r: {np.mean(skill_rewards):.3f} "
                  f"reward: {total_reward} ||| mean {np.mean(rewards):.3f} ||| max {np.max(rewards)} ||| {actions}")

            episodes += 1

            self.update(train_descriminator=bool((t + 1) % self.args.train_dc_interv == 0))



if __name__ == '__main__':
    env = CroudsorsingEnv()
    agent = VALOR()
    agent.train()