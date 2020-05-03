import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import copy

def weights_init_(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

class PreprocessLayers(nn.Module):
    def __init__(self):
        super(PreprocessLayers, self).__init__()
        self.inner = (nn.Conv2d(3, 1, kernel_size=2, stride=2, padding=0),
                       nn.Sigmoid(),
                       nn.Conv2d(1, 1, kernel_size=16, stride=16, padding=0),
                       nn.Sigmoid())

        self.layers_1 = nn.Sequential(*copy.deepcopy(self.inner))
        self.layers_2 = nn.Sequential(*copy.deepcopy(self.inner))
        self.layers_3 = nn.Sequential(*copy.deepcopy(self.inner))

        self.layers_2.load_state_dict(self.layers_1.state_dict())
        self.layers_3.load_state_dict(self.layers_1.state_dict())

        self.fc_size = 100

    def forward(self, state):
        pict1 = state[:, :, :, :320]
        pict2 = state[:, :, :, 320:640]
        pict3 = state[:, :, :, 640:]
        x1 = self.layers_1(pict1).view(-1, self.fc_size)
        x2 = self.layers_2(pict2).view(-1, self.fc_size)
        x3 = self.layers_3(pict3).view(-1, self.fc_size)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, context_dim):
        super(Actor, self).__init__()

        self.preprocess = PreprocessLayers()
        self.liner_context = nn.Linear(context_dim, 25)

        self.fc1 = nn.Linear(state_dim + 25, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.apply(weights_init_)

    def forward(self, state, context_one_hot):
        x = self.preprocess(state)
        context = self.liner_context(context_one_hot)

        x = torch.cat([x, context], dim=1)

        x_concat = x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x, x_concat

def weights_init_(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

class Critic(nn.Module):
    def __init__(self, state_dim, context_dim):
        super(Critic, self).__init__()
        self.preprocess = PreprocessLayers()
        self.liner_context = nn.Linear(context_dim, 25)

        self.fc1 = nn.Linear(state_dim + 25, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.apply(weights_init_)

    def forward(self, state, context_one_hot=None):
        x = self.preprocess(state)
        context = self.liner_context(context_one_hot)

        x = torch.cat([x, context], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, state_dim, input_dim, context_dim, hidden_dim=200, num_layers=1):
        super(Discriminator, self).__init__()
        self.preprocess = PreprocessLayers()
        self.liner_aaction = nn.Linear(3, 25)#3 = action_dim

        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=hidden_dim // 2, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_dim, context_dim)
        self.apply(weights_init_)
        nn.init.zeros_(self.linear.bias)

    def forward(self, state, action, context):
        x = self.preprocess(state)
        action = self.liner_aaction(action)

        state_action = torch.cat([x, action], dim=1)

        inter_states, _ = self.lstm(state_action.unsqueeze(0))
        logits_seq = self.linear(inter_states)
        logits = torch.mean(logits_seq, dim=1)

        policy = Categorical(logits=logits)
        label = policy.sample()
        log_context = policy.log_prob(context).squeeze()

        return label, log_context