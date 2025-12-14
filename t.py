import numpy as np

pos = np.array([1, 2, 3])

orient = np.array([[1.0, 0.0, 0.0], 
                    [0.0, -1.0, 0.0], 
                    [0.0, 0.0, -1.0]])

a = np.concatenate([pos, orient[0], orient[1], orient[2]])

print(a)