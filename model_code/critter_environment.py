from trajectory_processor import data_initializer

class Environment:
    def __init__():
        self.trajectory, self.data = data_initializer(DATA_FILE_NAME)
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
        self.current_index, self.final_index = self.trajectory()

    def state_and_reward(self, current_state, picked_action):
        """
        Should take the current state and the action and return the new state and the reward.
        """
        index = self.index_lookup[tuple(current_state)]
        if (index+1) > floor(self.data["end"]):
            next_state
