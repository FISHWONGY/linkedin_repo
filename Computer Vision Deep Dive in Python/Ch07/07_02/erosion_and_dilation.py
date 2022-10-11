import numpy as np
import matplotlib.pyplot as plt
import cv2
from EC_CV import *

# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams

rcParams["figure.figsize"] = (20, 8)

# Let's open a black and white picture

img = plt.imread("hi_there.bmp")
img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
print(np.shape(img))
plt.imshow(img, cmap="gray")

# Now let's perform a 3x3 erosion

kernel3 = np.ones((3, 3), np.uint8)
img2 = img
img2 = cv2.erode(img2, kernel3, iterations=1)
plt.imshow(img2, cmap="gray")
img3 = img2

# Now let's dilate it with a 5x5 kernel

kernel5 = np.ones((5, 5), np.uint8)
img2 = cv2.dilate(img2, kernel5, iterations=1)
plt.imshow(img2, cmap="gray")

# Now let's erode the dilated image

img2 = cv2.erode(img2, kernel5, iterations=1)
plt.imshow(img2, cmap="gray")

# Let's see the image before dilating and eroding, and after

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img3, cmap="gray")
ax[1].imshow(img2, cmap="gray")

# Let's open a different black and white picture

img = plt.imread("shapes.bmp")
print(np.shape(img))
plt.imshow(img, cmap="gray")

# Let's erode those shapes 4 times with a 3x3 kernel

img2 = img
img2 = cv2.erode(img2, kernel3, iterations=4)
plt.imshow(img2, cmap="gray")

# Now let's dilate those shapes twice with a 5x5 kernel

img2 = cv2.dilate(img2, kernel5, iterations=2)
plt.imshow(img2, cmap="gray")

# Now let's erode those shapes 8 times with a 5x5 kernel

img3 = img2
img2 = cv2.erode(img2, kernel5, iterations=8)
plt.imshow(img2, cmap="gray")

# Now let's dilate those shapes 8 times with a 5x5 kernel

img2 = cv2.dilate(img2, kernel5, iterations=8)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img3, cmap="gray")
ax[1].imshow(img2, cmap="gray")
