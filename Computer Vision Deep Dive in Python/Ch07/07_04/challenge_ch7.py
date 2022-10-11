import numpy as np
import matplotlib.pyplot as plt
import cv2
from EC_CV import *

# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

rcParams["figure.figsize"] = (17, 14)

# In this cell, we have the picture taken by the camera in the ceiling,
# as requested by the robot.
#
# Run this cell and move on to the next one.

img = plt.imread("warehouse.bmp")
plt.axis("off")
plt.imshow(img, cmap="gray")

# In this cell we have a special operation to create a color mask.
#
# This procedure is described in OpenCV's tutorials website:
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

# Convert RGB to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Define range of color in HSV
lower = np.array([10, 50, 50])
upper = np.array([90, 255, 255])

# Threshold the HSV image to get only brown colors
mask = cv2.inRange(hsv, lower, upper)

plt.imshow(mask, cmap="gray")

# Run this cell to appreciate the lingering white dots

rcParams["figure.figsize"] = (15, 14)
plt.imshow(mask[:300, 300:800], cmap="gray")

# Task #1: Get rid of the noise
# In this cell, use morphological transformations to get rid of the
# white dots throughout the mask.

rcParams["figure.figsize"] = (17, 14)
# Write your code here

# Task #2: Make the blobs grow.
# In this cell, use morphological transformations to make the obstacle
# blobs grow.
# It's simple: Just dilate the mask with a 5x5 kernel about 10 times
# or with a 3x3 kernel about 20 times.

# Write your code here

"""## Now run all cells again, but skip task #1 to see what happens"""
