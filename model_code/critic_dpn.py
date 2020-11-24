from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class DPN(nn.Module):
    def __init__(self, input_size):
        super(DPN, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.fc41 = nn.Linear(input_size, input_size)
        self.fc42 = nn.Linear(input_size, input_size)
        self.fc51 = nn.Linear(input_size, input_size)
        self.fc52 = nn.Linear(input_size, input_size)
        self.fc61 = nn.Linear(input_size, input_size)
        self.fc62 = nn.Linear(input_size, input_size)

    def forward(self, x):
        residual = x
        h = F.relu(self.fc1(x)) + x
        h = F.relu(self.fc2(h)) + h
        h = F.relu(self.fc3(h)) + h
        #Mu side
        mu = F.relu(self.fc41(h)) + h
        mu = F.relu(self.fc51(mu)) + mu
        mu = self.fc61(mu)
        #logsigma side
        logvar = F.relu(self.fc42(h))
        logvar = F.relu(self.fc52(logvar))
        logvar = self.fc62(logvar)
        return mu, logvar


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.fc4 = nn.Linear(input_size, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x)) + x
        h = F.relu(self.fc2(h)) + h
        h = F.relu(self.fc3(h)) + h
        out = -self.fc4(h)
        return out
