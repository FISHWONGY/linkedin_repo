import numpy as np
import matplotlib.pyplot as plt
import cv2
from EC_CV import *

# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

rcParams["figure.figsize"] = (20, 8)

# Let's open a black and white picture

img = plt.imread("shapes_n_dots.bmp")
print(np.shape(img))
plt.imshow(img, cmap="gray")

# Let's zoom in to see two special shapes

plt.imshow(img[200:, 300:510], cmap="gray")

# Now let's perform a 3x3 open

kernel3 = np.ones((3, 3), np.uint8)

img2 = img
img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel3)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap="gray")
ax[1].imshow(img2, cmap="gray")

# Now let's perform a 3x3 close

img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel3)
plt.imshow(img2, cmap="gray")

"""# Now let's work with the 5 x 5 kernel """

# Let's open once
kernel5 = np.ones((5, 5), np.uint8)
img2 = img
img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel5)
plt.imshow(img2, cmap="gray")

# Now let's perform a 5x5 close

img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel5)
plt.imshow(img2, cmap="gray")

# Let's zoom in to see two special shapes

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img[200:, 300:510], cmap="gray")
ax[1].imshow(img2[200:, 300:510], cmap="gray")

# Now let's perform a 5x5 close first
img2 = img
img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel5)

# And *then* a 5x5 open
img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel5)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img[200:, 300:510], cmap="gray")
ax[1].imshow(img2[200:, 300:510], cmap="gray")
