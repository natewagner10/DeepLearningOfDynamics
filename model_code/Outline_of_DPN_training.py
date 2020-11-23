"""
WARNING:

Code will not be executable from this alone. Must clean up the TODOs.

Provides a backbone for the training method on a deep policy network.

Presumably, the input to the network will be images as the current state (including current resource usage, and random M jobs to schedule).
The output predictions are the probabilities for choosing to schedule one of the M jobs.

Most important is the train_on_jobs method for updating the weights of the network, in accordance with DPN loss functions.
"""

from critic_dpn import DPN, Critic

import numpy as np
from random import random
from scipy.stats import norm
from critter_environment import Environment

from torch.distributions import Independent, Normal
#from torch.distributions.independent import Independent
#from torch.distributions.normal import Normal
import torch

DATA_FILE_NAME = "trajectory_dict.pickle"
TRAJECTORY_LENGTH = 30 #Approximately 10 seconds
MIDPOINTS = 2 #splits video data into trajectories of length above, but this determines the amount of overlap across trajectories

EPSILON_PERTURBATIONS = False  #if we want the network to predict how to perturb LS vector.
ITERATIONS = 1000 #kinda like epochs?
BATCH_SIZE = 10   #Might be the exact same thing as episodes, up for interpretation.
EPISODES = 20     #How many trajectories to explore for a given job. Essentually to get a better estimate of the expected reward.
DISCOUNT = 0.99   #how much to discount the reward
ALPHA = 0.001     #learning rate?

def curried_valuation(length_of_longest_trajectory):
    '''
    Given the length of the longest trajectory of a set of episodes;

    returns the function that will compute the valuation of an episode array (while padding it)

    Result intended to be used as  map(valuation, episodes_array) to return valuation of each episodes.
    '''
    def valuation(episode):
        '''
        returns the valuation of an episode (with padding)
        input: [(s_0, a_0, r_0), ... ,(s_t, a_t, r_t)]         potentially t<L
        output: [v_0, v_1, ... v_L]
        '''
        x = len(episode)
        if x != length_of_longest_trajectory:
            #If the episode isn't as long as the longest trajectory, pad it
            episode.extend([(0,0,0) for x in range(length_of_longest_trajectory-x)]) #have to make sure the numbers line up correctly
        # TODO
        #compute valuation with the episode/trajectory after it's been padded. There could be something clever here.
        out = np.array([valuation for valuation in range(length_of_longest_trajectory)])
        #out = do_the_thing(episode)
        return out
    return valuation

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)



class DpnTraining:
    def __init__(self, INPUT_SIZE):
        '''
        INPUT_SIZE = size and shape from the environment's output for a state  TODO
        OUTPUT_SIZE = number of possible actions                               TODO

        Probably include stuff to interact with the environment after inputting a class

        all caps words are hyperparameters you would set.
        '''
        self.env = Environment(DATA_FILE_NAME, EPSILON_PERTURBATIONS = EPSILON_PERTURBATIONS)

        # Define the network
        self.network = DPN(INPUT_SIZE)
        self.network.apply(weights_init_uniform_rule)
        self.critic = Critic(INPUT_SIZE)


    def train(self, ITERATIONS):
        optimizer = torch.optim.Adam(self.network.parameters(), lr = 3e-3) #This is roughly based on some pytorch examples. We use this to update weights of the model.
        for i in range(ITERATIONS):
            first_frame = self.env.make_start_state() #this would be a list of starting states
            jobs = [first_frame] #TODO: Coerce job variable to appropriate pytorch type. Necessary due to environment not set up to handle processing different trajectories.
            self.train_on_jobs(jobs, optimizer)


    def forward(self, state):
        '''
        The forward pass of the network on the given state. Returns the output probabilites for taking the OUTPUT_SIZE probabilites

        might already be defined from the initialization after defining your model
        '''
        state = state.astype(np.float)
        state = torch.from_numpy(state).float()
        print(state)
        mu, sigma = self.network(state)
        return mu, sigma


    def trajectory(self, current_state, refresh_defaults = True, output_history = []):
        '''
        Maybe this implementation doesn't utilize GPUs very well, but I have no clue or not.

        Final output looks like:
        [(s_0, a_0, r_0), ..., (s_L, a_L, r_l)]
        '''
        if refresh_defaults:
            output_history = []
        mu, log_sigma = self.forward(current_state)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
        sigma = torch.exp(log_sigma)                                                             #This might not work, please see this pull request?  https://github.com/pytorch/pytorch/pull/11178
        distribution = Independent(Normal(mu, sigma),1)
        picked_action = distribution.sample() #TODO must be redone to work with pytorch.
        new_state, reward = self.env.state_and_reward(current_state, picked_action) #Get the reward and the new state that the action in the environment resulted in. None if action caused death. TODO build in environment
        output_history.append( (current_state, picked_action, reward) )
        if new_state is None: #essentially, you died or finished your trajectory
            return output_history
        return  self.trajectory(new_state, False, output_history)

    def train_on_jobs(self,jobset, optimizer):
        '''
        Training from a batch. Kinda presume the batch is a set of starting states not sure how you have the implemented states (do they include actions internally?)

        example shape of episode_array
        [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3]
        ]
        '''
        #optimizer.zero_grad()#Basically start gradient or how you'll change weights out at 0 but with the shape or whatever you need to update the weights through addition. TODO figure out how this thing should look
        for job_start in jobset:
            optimizer.zero_grad()
            #episode_array is going to be an array of length N containing trajectories [(s_0, a_0, r_0), ..., (s_L, a_L, r_0)]
            episode_array = [self.trajectory(job_start) for x in range(EPISODES)]
            # Now we need to make the valuations
            longest_trajectory = max(len(episode) for episode in episode_array)
            valuation_fun = curried_valuation(longest_trajectory)
            cum_values = np.array(map(episode_array, valuation_fun)) #should be a EPISODESxlength sized
            #can compute baselines without a loop?
            baseline_array = np.array([sum(cum_values[:,i])/EPISODES for i in range(longest_trajectory)]) #Probably defeats the purpose of numpy, but we're essentially trying to sum each valuation array together, and then divide by the number of episodes TODO make it work nicely
            for i in range(EPISODES): #swapped two for loops
                for t in range(longest_trajectory):
                    try:
                        state, action, reward = episodes_array[i][t]
                    except IndexError: #this occurs when the trajectory died
                        break
                    #first two products are scalars, final is scalar multiplication of computed gradients on the NN
                    mu, log_sigma = self.forward(state)
                    sigma = torch.exp(log_sigma)
                    distribution = Independant(Normal(mu, sigma),1) #Defines a pytorch distribution equivalent to MultivariateNormal distribution with sigma as the diagonal.
                    if i ==0 and t == 0:
                        loss = -(cum_values[i][t]-baseline_array[t])*ALPHA*distribution.log_prob(action) #This is what it should look like in pytorch. Added negative on recommendation of pytorch documentation
                    else:
                        loss += -(cum_values[i][t]-baseline_array[t])*ALPHA*distribution.log_prob(action)
            loss.backward() #Compute the total cumulated gradient thusfar through our big-ole sum of losses
            optimizer.step() #Actually update our network weights. The connection between loss and optimizer is "behind the scenes", but recall that it's dependent
