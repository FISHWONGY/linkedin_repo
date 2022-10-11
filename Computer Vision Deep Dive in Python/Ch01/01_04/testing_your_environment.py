# Now let's check your Python version

import sys

print("Python version", sys.version)

# Now check your version of Numpy and create a matrix

import numpy as np

print("Numpy version", np.version.version)
m = np.array([[1, 2], [3, 4]])
print(m)

# Commented out IPython magic to ensure Python compatibility.
# Now let's try opening an image and displaying it with Matplotlib:

import matplotlib.pyplot as plt

# %config InlineBackend.figure_formats = 'retina'

img = plt.imread("komodo.jpg")
plt.axis("off")
plt.imshow(img)

# Finally, let's open an image using OpenCV
import cv2

print("OpenCV version", cv2.__version__)
image = cv2.imread("./linkedin_repo/Computer Vision Deep Dive in Python/Ch01/01_04/komodo.jpg")
cv2.imshow("OpenCV", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""# You're good to go!"""
