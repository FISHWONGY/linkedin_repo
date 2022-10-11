# Commented out IPython magic to ensure Python compatibility.
# Let's open the Playspace image again

import numpy as np
import matplotlib.pyplot as plt

# %config InlineBackend.figure_formats = 'retina'
# %matplotlib inline
from EC_CV import *
from matplotlib import rcParams

plt.rcParams["figure.figsize"] = (8, 8)

toys = adapt_PNG(plt.imread("playspace.png"))
plt.axis("off")
plt.imshow(toys)

rcParams["figure.figsize"] = (20, 8)

# Calculate regular average and weighted average
toys_avg = np.dot(toys[..., :3], [1 / 3, 1 / 3, 1 / 3])
toys_wgt = np.dot(toys[..., :3], [0.299, 0.587, 0.114])

# display images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(toys_avg, cmap="gray")
ax[1].imshow(toys_wgt, cmap="gray")

# Now let's try it on another picture

fruit = plt.imread("fruit.jpg")
plt.imshow(fruit)

fruit_avg = np.dot(fruit[..., :3], [1 / 3, 1 / 3, 1 / 3])
fruit_wgt = np.dot(fruit[..., :3], [0.299, 0.587, 0.114])

# figure size in inches
rcParams["figure.figsize"] = 20, 8

# display images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(fruit_avg, cmap="gray")
ax[1].imshow(fruit_wgt, cmap="gray")
