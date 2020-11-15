"""
WARNING:

Code will not be executable from this alone. Must clean up the TODOs.

Provides a backbone for the training method on a deep policy network.

Presumably, the input to the network will be images as the current state (including current resource usage, and random M jobs to schedule).
The output predictions are the probabilities for choosing to schedule one of the M jobs.

Most important is the train_on_jobs method for updating the weights of the network, in accordance with DPN loss functions.
"""


import numpy as np
from random import random
from scipy.stats import norm
from critter_environment import Environment


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

def log_likelihood(mu,sigma,activation):
    '''
    Calculation of the log-likelihood of an arbitrary k-dimensional diagonal gaussian distribution:

    I.e., log(\pi_theta(a|s))

    HOWEVER, this can likely be fixed with some PURE pytorch functionality!
    '''
    k = len(mu)
    out = -0.5*sum((np.square(mu-activation))/np.square(sigma) + 2*np.log(sigma)) + k*np.log(2*np.pi)
    return out


def randomly_selected_action(mu_and_sigma, slice = False):
    '''
    Recall that we'll be utilizing the "reparameterization trick" for the output vector of our network.

    Converts the output of our network [mu_0, sigma_0, ..., mu_M, sigma_M] into:

    1. A randomly selected perterbation amount: [e_0, e_2, ..., e_M] by letting e_i = mu_i + sigma_i * z_i where z_i ~ N(0,1)
    2. Likilihoods of having drawn each epsilon from e_i ~ N(mu_i, sigma_i), by calculating pdf(z_i, mu=0, sigma=1), in R this is dnorm(z_i, 0, 1)
    '''
    epsilon_length = int(len(mu_and_sigma)/2)
    if len(mu_and_sigma)%2 == 1:
        raise ValueError("Oh, actual fuck, we somehow made our output layer 'oddly' sized.")
    #First we get the mean vector and sigma vector from our model's output.
    if slice:
        mean = np.array(mu_and_sigma[:epsilon_length])
        sigma = np.array(mu_and_sigma[epsilon_length:])
    else:
        mean = np.array([y for x,y in enumerate(mu_and_sigma) if x%2 == 0])
        sigma = np.array([y for x,y in enumerate(mu_and_sigma) if x%2 == 1])
    sigma = np.exp(sigma)
    #Select M/2 total IID z-stats from N(0,1)
    z_stat = np.random.normal(size = epsilon_length)
    #Calculate likelihoods of having pulled each epsilon
    likelihoods = np.array([norm.pdf(x) for x in z_stat])
    #Calculate our actual perterbation amound, epsilon
    epsilon = mean + sigma*z_stat
    return epsilon, likelihoods

"""
This code is if you're action space is finite, and the network outputs are probabilities instead of the parameters of our distribution.


def randomly_selected_action(probs):
    '''
    Given a set of output probabilities corresponding to actions (or jobs to schedule):

    Randomly select one action with the described probabilities.

    Output the index of the job to schedule and the probability that we chose that action.

    For large probability lists, we might choose the final probability less than we expect due to floating point arithmetic problems.
    '''
    action_number = np.random.random()
    index = 0
    lower = 0
    upper = probs[index]
    while not (lower <= action_number and action_number <= upper):
        index += 1
        lower += upper
        upper += probs[index]
    return index, probs[index]
"""


class DPN:#(keras_module or whatever):
    def __init(self)__:
        #Super(self, __init__) #Initialize base methods of keras NN module stuff?
        '''
        define your shit about the NN stuff initial weights, architecture, and so forth
        WE NEED TO FIGURE OUT HOW TO GET GRADIENT of log(policy(state, action))
        Also allow weights to be updated through addition? Something like that.


        INPUT_SIZE = size and shape from the environment's output for a state  TODO
        OUTPUT_SIZE = number of possible actions                               TODO

        Probably include stuff to interact with the environment after inputting a class

        all caps words are hyperparameters you would set.
        '''
        self.env = Environment()
    def train(self, ITERATIONS):
        optimizer = optim.Adam(self.model.parameters(), lr = 3e-3) #This is roughly based on some pytorch examples. We use this to update weights of the model.
        for i in range(ITERATIONS):
            first_frame, last_fram = self.env.make_start_state() #this would be a list of starting states
            jobs = [first_frame] #due to environment not set up to handle processing different trajectories.
            self.train_on_jobs(jobs, optimizer)


    def forward(self, state):
        '''
        The forward pass of the network on the given state. Returns the output probabilites for taking the OUTPUT_SIZE probabilites

        might already be defined from the initialization after defining your model
        '''
        pass


    def trajectory(self, current_state, refresh_defaults = True, output_history = []):
        '''
        Maybe this implementation doesn't utilize GPUs very well, but I have no clue or not.

        Final output looks like:
        [(s_0, a_0, r_0), ..., (s_L, a_L, r_l)]
        '''
        if refresh_defaults:
            output_history = []
        mu_and_sigma = self.forward(current_state)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
                                                                     #This might not work, please see this pull request?  https://github.com/pytorch/pytorch/pull/11178
        picked_action, likelihoods = randomly_selected_action(mu_and_sigma) #TODO must be redone to work with pytorch.
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
                    mu, sigma = self.forward(state)
                    distribution = Independant(Normal(mu, sigma),1) #Defines a pytorch distribution equivalent to MultivariateNormal distribution with sigma as the diagonal.
                    if i ==0 and t == 0:
                        loss = -(cum_values[i][t]-baseline_array[t])*ALPHA*distribution.log_prob(action) #This is what it should look like in pytorch. Added negative on recommendation of pytorch documentation
                    else:
                        loss += -(cum_values[i][t]-baseline_array[t])*ALPHA*distribution.log_prob(action)
            loss.backward() #Compute the total cumulated gradient thusfar through our big-ole sum of losses
            optimizer.step() #Actually update our network weights. The connection between loss and optimizer is "behind the scenes", but recall that it's dependent
