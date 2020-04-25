import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def weights_init_(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

class Actor(nn.Module):
    def __init__(self, state_dim, context_dim, load_path=None):
        super(Actor, self).__init__()
        self.preprocess = nn.Sequential(nn.Conv2d(3, 5, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(10, 10, padding=1),
                                        nn.Conv2d(5, 10, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(10, 10, padding=1),
                                        nn.Conv2d(10, 15, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(5, 5, padding=1))
        if load_path is not None:
            self.preprocess.load_state_dict(torch.load(load_path))
        self.fc_size = state_dim
        self.fc1 = nn.Linear(state_dim + context_dim, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 3)
        self.apply(weights_init_)

    def forward(self, x, context_one_hot=None):
        x = self.preprocess(x)
        x = x.view(-1, self.fc_size)
        x = torch.cat([x, context_one_hot], dim=1)
        x_concat = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x, x_concat

def weights_init_(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

class Critic(nn.Module):
    def __init__(self, state_dim, context_dim, load_path=None):
        super(Critic, self).__init__()
        self.preprocess = nn.Sequential(nn.Conv2d(3, 5, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(10, 10, padding=1),
                                        nn.Conv2d(5, 10, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(10, 10, padding=1),
                                        nn.Conv2d(10, 15, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(5, 5, padding=1))
        if load_path is not None:
            self.preprocess.load_state_dict(torch.load(load_path))
        self.fc_size = state_dim
        self.fc1 = nn.Linear(state_dim + context_dim, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 1)
        self.apply(weights_init_)

    def forward(self, x, context_one_hot=None):
        x = self.preprocess(x)
        x = x.view(-1, self.fc_size)
        x = torch.cat([x, context_one_hot], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, state_dim, input_dim, context_dim, hidden_dim=32, num_layers=1):
        super(Discriminator, self).__init__()
        self.preprocess = nn.Sequential(nn.Conv2d(3, 5, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(10, 10, padding=1),
                                        nn.Conv2d(5, 10, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(10, 10, padding=1),
                                        nn.Conv2d(10, 15, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(5, 5, padding=1))
        self.fc_size = state_dim
        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=hidden_dim // 2, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_dim, context_dim)
        self.apply(weights_init_)
        nn.init.zeros_(self.linear.bias)

    def forward(self, state, action, context):
        x = self.preprocess(state)
        x = x.view(-1, self.fc_size)
        state_action = torch.cat([x, action], dim=1)
        inter_states, _ = self.lstm(state_action.unsqueeze(0))
        logits_seq = self.linear(inter_states)
        logits = torch.mean(logits_seq, dim=1)

        policy = Categorical(logits=logits)
        label = policy.sample()
        log_context = policy.log_prob(context).squeeze()

        return label, log_context