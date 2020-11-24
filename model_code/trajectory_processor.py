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
import torch
import numpy as np
#from pandas import read_csv
"""
kinda_cursed is definitely cursed - no question about it - but it provides considerable speed benefits, and completely generalizes situations where there isn't a single long path through a video (multiple clips, for example).
We encode the connections between frames as a dictionary, and the function allows you to find n frames away from x (returning the last frame m < n if there are not n more frames after x).
If x is not a valid frame (or there is no subsequent, return None).

Used as:

kinda_cursed(n)(x) = nth frame after x

Just a bit of exposition on the subesequent code:

(lambda func: lambda x: lambda y: func(x,y) )
This takes any function, f, that takes two arguments and curries it, i.e., returns a function g, such that f(x, y) = g(x)(y)

(lambda r: lambda n, x: r(r, n, x, n))
This takes a (carefully crafted) function whose first argument is implicitly a reference to itself, and returns a version of only two arguments that makes the first argument a reference to itself. Essentially allowing for completely anonymous recursive functions (defining functions has significant overhead). This presumes the passed in function is using the first argument as self reference.

lambda s, n, x, no:
  ssframe[x] if n == 1 and x in ssframe else(
    s(s, n-1, ssframe[x], n) if x in ssframe else (
        x if n != no else None
    )
  )
Carefully crafted function that performs ssframe[x] (subsequent frame of x), n number of times. If starting x is not a valid frame, returns none. If some mth frame has no subsequent frame, returns that frame.
"""

def data_initializer(file_name, trajectory_length = 30, midpoints = 2):
    """
    Preprocess the data and return a generating function that'll be utilized by the environment.
    """
    import pickle
    data = pickle.load(open(file_name, "rb"))
    #Make the subsequent frame dictionary
    tensor_convert = lambda x: torch.from_numpy(x.astype(np.float)).float()
    data = {key:tensor_convert(value) for key, value in data.items() }
    keys = list(data.keys())
    keys.sort()
    lookup = {tuple(y.numpy()):x for x,y in data.items()}
    subsequent = dict()
    for i in range(len(keys)):
        if i != 0:
            key = keys[i-1]
            value = keys[i]
            subsequent[key] = value
    '''kinda_cursed = (lambda func: lambda x: lambda y: func(x,y) )(
  (lambda r: lambda n, x: r(r, n, x, n))(
    lambda s, n, x, no: subsequent[x] if n == 1 and x in subsequent else( s(s, n-1, subsequent[x], n) if x in subsequent else (x if n != no else None) ) )
    )(trajectory_length)'''
    def not_as_cursed(n,x):
        current = x
        for i in range(n):
            if current in subsequent:
                current = subsequent[current]
            elif i != 0 and current not in subsequent:
                return current
            else:
                return None
        return current
    kinda_cursed = (lambda f: lambda n: lambda x: f(n,x))(not_as_cursed)(trajectory_length)
    valid_start = keys[::int(round(trajectory_length/midpoints))]
    random_start = list(random.sample(valid_start, len(valid_start)))
    def training_generator(starts = random_start, backup = valid_start):
        if starts == []:
            starts.extend(random.sample(backup, len(backup)))
        start_frame = starts.pop()
        end_frame = kinda_cursed(start_frame)
        return start_frame, end_frame
    return data, lookup, subsequent, training_generator




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
'''

if __name__ == "__main__":
    FILE = ""
    import pickle
    d = pickle.loads(FILE)
    for key, value in d.items():
        if type(key) is not int:
            print(key, value)
