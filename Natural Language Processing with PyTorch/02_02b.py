import torch
import numpy as np

# Construct a tensor from an array
array = [[1, 2], [7, 4], [5, 6]]
tensor0 = 
print(tensor0)
print("The data structure type of tensor0: ", type(tensor0))
print("The data type of tensor0: ", tensor0.dtype)
print("The shape of tensor0: ", tensor0.shape)

# Construct a tensor from a numpy array
np_array = np.array([[1, 2], [7, 4], [5, 6]])
tensor1 = 
print(tensor1)

"""**b. Common Methods: Slicing and Concatenation**

*Slicing*
"""

tensorA = torch.tensor([[1, 1, 1], [2, 2, 2]])
tensorB = torch.tensor([[3, 3, 3], [4, 4, 4]])

# Slicing is all the same as numpy arrays
print('Slicing the first two rows of tensorA (index one inclusive index two exclusive): ')
print()
print('Slicing the first two columns of tensorA (take all rows, then slice columns): ')
print()

"""*Concatenation*"""

print('Vertically concatenate tensorA and tensorB: (default: dim=0)')
concat_v = 
print(concat_v)

print('Horizontally concatenate tensorA and tensorB: (dim=1)')
concat_h = 
print(concat_h)