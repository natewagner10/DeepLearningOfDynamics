import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Independent, Normal


class DPN_alt(nn.Module):
    """AGENTS"""
    def __init__(self, alpha, input_size):
        super(DPN, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.fc61 = nn.Linear(input_size, input_size)
        self.fc62 = nn.Linear(input_size, input_size)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        residual = x
        h = T.tanh(self.fc1(x)) + x
        h = T.tanh(self.fc3(h)) 
        #Mu side
        mu = F.leaky_relu(self.fc61(h)) + residual
        #logsigma side
        logvar = F.leaky_relu(self.fc62(logvar))
        return mu, logvar

class DPN(nn.Module):
    """AGENTS"""
    def __init__(self, alpha, input_size):
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
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        residual = x
        h = F.tanh(self.fc1(x)) + x
        h = F.tanh(self.fc2(h)) + h + x
        h = F.tanh(self.fc3(h)) 
        #Mu side
        mu = F.tanh(self.fc41(h)) + h
        mu = F.tanh(self.fc51(mu)) + h
        mu = F.leaky_relu(self.fc61(mu)) + residual
        #logsigma side
        logvar = F.leaky_relu(self.fc42(h))
        logvar = F.leaky_relu(self.fc52(logvar))
        logvar = F.leaky_relu(self.fc62(logvar))
        return mu, logvar


class Critic(nn.Module):
    """CRITIC"""
    def __init__(self, beta, input_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.fc4 = nn.Linear(input_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h)) 
        out = -1*self.fc4(h)
        return out

class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99):
        self.gamma = gamma
        self.log_probs = None
        self.alpha = alpha
        self.beta = beta
        self.actor = DPN(alpha, input_dims)
        self.critic = Critic(beta, input_dims)
        self.critic_target = Critic(beta, input_dims)
        self.update_critic_target()
    
    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic_target.state_dict())

    def choose_action(self, observation):
        mu, sigma  = self.actor.forward(observation)#.to(self.actor.device)
        sigma = T.exp(sigma)
        action_probs = Independent(Normal(mu, sigma),1)
        probs = action_probs.sample()
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        return probs

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic_target.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = ((reward + self.gamma*critic_value_*(1-int(done))) - \
                                                                critic_value)

        critic_loss = delta**2
        critic_loss.backward(retain_graph = True)
        #(actor_loss + critic_loss).backward()
        #T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.002)
        #T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        #print(critic_value)
        self.critic.optimizer.step()
        actor_loss = -1*self.log_probs * delta.detach()
        actor_loss.backward()
        self.actor.optimizer.step()
        