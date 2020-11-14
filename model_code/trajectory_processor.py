"""
ProjectDirectory:
--|Trajectory
  --traj1.csv
  --traj2.csv
  .
  .
  .
  --trajn.csv

"""

import os
import random
#from pandas import read_csv
'''
def trajectory_grabber_initializer(absolute_path_to_trajectory_folder: str, file_prefix: str, file_suffix: str, frame_skip: int, frame_num_col: str = "frame_num") -> function:
    """
    Takes absolute path to trajectory folder, file prefixes in the directory, file suffix and the number of frames that are skipped across all trajectories.

    Returns the formatted data grabber that doesn't have to preprocess anything.
    """
    x = len(os.listdir(absolute_path_to_trajectory_folder))
    random_csv_order = random.sample(range(x), x)
    def formatted_data_grabber(trajectory_directory_order = random_csv_order, traj_order = random_csv_order.copy(), frame_skip = frame_skip):
        if trajectory_directory_order == []:
            new_epoch = random.sample(traj_order, x)
            trajectory_directory_order.extend(new_epoch)
        #get the trajectory
        trajectory_number = trajectory_directory_order.pop()
        #load that trajectory into pandas dataframe
        file_name = file_prefix+ str(trajectory_number) + file_suffix
        file_name = os.path.join(absolute_path_to_trajectory_folder, file_name)
        df = read_csv(file_name)
        # get the first frame number
        first = df[0,1]
        #convert dataframe as dicitonary of
        data_as_dict = df.set_index(frame_num_col).T.to_dict('list').itmes()
        output_dictionary = dict((x,np.array(y)) for x,y in data_as_dict )
        return None
    return formatted_data_grabber

print(type(trajectory_grabber_initializer("poo", "zoo", "zoo", 5)))
'''


def data_initializer(file_name, trajectory_length = 30):
    """
    Preprrocesses some data and returns a generating function that spits out start and end frame numbers.

    Assumes no frame skipping in the loaded dictionary.
    """
    import pickle
    data = pickle.loads(file_name)
    data["global start"] = data["start"]
    data["global end"] = data["end"]
    increment = data["increment"]
    start_frames = [range(data["global start"], data["global end"]-1, floor(trajectory_length/2))]
    size = len(start_frames)
    initial_points = random.sample(start_frames, size)
    def training_generator(data = data, x = initial_points, x_copy = initial_points.copy()):
        if x == []:
            new_epoch = random.sample(x_copy, size)
            x.extend(new_epoch)
        first_frame = x.pop()
        last_frame = first_frame + trajectory_length
        data['start'], data['end'] = first_frame, last_frame
        return first_frame, last_frame
    return training_generator, data


if __name__ == "__main__":
    FILE = ""
    import pickle
    d = pickle.loads(FILE)
    for key, value in d.items():
        if type(key) is not int:
            print(key, value)
