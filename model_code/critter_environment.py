from trajectory_processor import data_initializer
import numpy as np


class Environment:
    def __init__(self, DATA_FILE_NAME, EPSILON_PERTURBATIONS = False):
        self.trajectory, self.data = data_initializer(DATA_FILE_NAME) #TODO: Needs some modifications inn trajeectory Processor
        self.index_lookup = dict((tuple(y),x) for x,y in self.data.items())
        if len(self.index_lookup) != len(self.data):
            print("Warning there are some conflicts in the data lookup")
            print("Attempting to correct this problem")
            self.index_lookup = dict()
            for x, y in self.data.items():
                y = tuple(y)
                if y in self.index_lookup:
                    if type(self.index_lookup[y]) is int:
                        self.index_lookup[y] = [self.index_lookup[y]]
                    index_lookup[y].append(x)
                else:
                    self.index_lookup[y] = x

    def make_start_state(self):
        """
        TODO: Look at how trajectory processor handles this. Important that if we go with the subsequent dictionary, then we'll be grabbing actually relvant frames.
        """
        self.start_index, self.final_index = self.trajectory()
        return self.data[self.start_index]

    def state_and_reward(self, current_state, picked_action):
        """
        Should take the current state and the action and return the new state and the reward.
        """
        index = self.index_lookup[tuple(current_state)]
        if type(index) is list:
            for i in index:
                if i < self.final_index and i > self.start_index:
                    index = i
                    break
        next_index = self.next_index_lookup.setdefault(index, None)
        if (next_index) == floor(self.data["end"]) or next_index == None: #TODO subsequent element index
            if next_index == None:
                raise("Reached end, end, end of trajectory without escaping first!")
            else:
                returned_state = None
                next_state = self.data[next_index]
        else:
            next_state = self.data[next_index]
            returned_state = next_state
        #now calculate reward
        if EPSILON_PERTURBATIONS:
            guess = current_state + picked_action
        else:
            guess = picked_action
        return returned_state, reward
