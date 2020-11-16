from trajectory_processor import data_initializer
import numpy as np


class Environment:
    """

    """
    def __init__(self, DATA_FILE_NAME, trajectory_length = 30, midpoints = 2, EPSILON_PERTURBATIONS = False):
        self.epsilon_pert = EPSILON_PERTURBATIONS
        self.data, self.index_lookup, self.subsequent, self.trajectory = data_initializer(DATA_FILE_NAME, trajectory_length = trajectory_length, midpoints = midpoints)
        if len(self.index_lookup) != len(self.data):
            print("Warning there are some conflicts in the data lookup")
            print("Attempting to correct this problem")
            raise("Just kidding, please edit the environment to fix this problem. I've removed it due to our data having nice qualities. Turns out I was wrong.")
            for x, y in self.data.items():
                y = tuple(y)
                if y in self.index_lookup:
                    if type(self.index_lookup[y]) is int:
                        self.index_lookup[y] = [self.index_lookup[y]]
                    self.index_lookup[y].append(x)
                else:
                    self.index_lookup[y] = x

    def make_start_state(self):
        """Sets implicit start_index and final_index for some number of trajectories"""
        self.start_index, self.final_index = self.trajectory()
        return self.data[self.start_index]

    def state_and_reward(self, current_state, picked_action):
        """
        Should take the current state and the action and return the new state and the reward.
        """
        index = self.index_lookup[tuple(current_state)]
        next_index = self.subsequent[index]
        next_state = self.data[next_index]
        if next_index == self.final_index:
            returned_state = None
        else:
            returned_state = next_state
        #now calculate reward
        if self.epsilon_pert:
            guess = current_state + picked_action
        else:
            guess = picked_action
        reward = -np.linalg.norm(guess - next_state)
        return returned_state, reward

if __name__ == "__main__":
    history = []
    DATA_FILE_NAME = "trajectory_dict.pickle"
    env = Environment(DATA_FILE_NAME, True)
    for i in range(1000):
        start = env.make_start_state()
        for j in range(5):
            current_state = start
            while current_state is not None:
                x = np.zeros(20)
                current_state, reward = env.state_and_reward(current_state,x)
                history.append(reward)

    print(history)
