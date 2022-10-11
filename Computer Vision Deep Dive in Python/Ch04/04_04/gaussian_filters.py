# Commented out IPython magic to ensure Python compatibility.
# Let's work with the picture of a house

import numpy as np
import matplotlib.pyplot as plt

# %config InlineBackend.figure_formats = 'retina'
from matplotlib import rcParams
import cv2

rcParams["figure.figsize"] = 20, 8

img = plt.imread("house.jpg")
plt.axis("off")
plt.imshow(img)

# OpenCV's GaussianBlur function works with color and grayscale images

blurred = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
plt.imshow(blurred)

# Now let's see all 3 blur filters at once

rcParams["figure.figsize"] = 18, 14
kernel = np.ones((15, 15), np.float32) / 225
blur1 = cv2.medianBlur(img, 15)
blur2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
blur3 = cv2.GaussianBlur(img, (15, 15), cv2.BORDER_DEFAULT)
titles = ["Original Image", "Median Filter", "Average Filter", "Gaussian Filter"]
images = [img, blur1, blur2, blur3]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# Let's zoom in to see the main entrance.

rcParams["figure.figsize"] = 20, 12

fig, ax = plt.subplots(1, 2)
ax[0].imshow(blur2[200:550, 150:450])
ax[1].imshow(blur3[200:550, 150:450])
