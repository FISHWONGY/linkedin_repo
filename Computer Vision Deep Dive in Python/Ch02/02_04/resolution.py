import numpy as np
import matplotlib.pyplot as plt

# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

plt.rcParams["figure.figsize"] = (10, 8)  # (Width, Height) supposedly in inches

img1 = plt.imread("dog800x600.jpg")

plt.imshow(img1)

img2 = plt.imread("dog300x225.jpg")
plt.imshow(img2)

img3 = plt.imread("dog120x90.jpg")
plt.imshow(img3)

plt.rcParams["figure.figsize"] = (1, 1)  # (Width, Height) supposedly in inches
plt.imshow(img3)
