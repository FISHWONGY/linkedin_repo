import numpy as np
import matplotlib.pyplot as plt

# %config InlineBackend.figure_formats = 'retina'
from EC_CV import *
from matplotlib import rcParams

plt.rcParams["figure.figsize"] = (8, 8)

toys = adapt_PNG(plt.imread("playspace.png"))
plt.imshow(toys)
np.shape(toys)

# Let's calculate the average for each pixel


def RGB_to_grayscale(RGB_pic):
    rows, cols, temp = np.shape(RGB_pic)
    gs = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            gs[i, j] = np.average(RGB_pic[i, j])
    return gs


toys_gs = RGB_to_grayscale(toys)
plt.imshow(toys_gs, cmap="gray")
np.shape(toys_gs)

# Took too long? Let's try a compact expression

toys_gs = np.dot(toys[..., :3], [1 / 3, 1 / 3, 1 / 3])
plt.imshow(toys_gs, cmap="gray")

# Now let's try it on a big picture

fruit = plt.imread("fruit.jpg")
plt.imshow(fruit)
np.shape(fruit)

fruit_gs = np.dot(fruit[..., :3], [1 / 3, 1 / 3, 1 / 3])
plt.imshow(fruit_gs, cmap="gray")
