# Commented out IPython magic to ensure Python compatibility.
# First, let's create a simple 3x3 image

import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = 'retina'

img = np.array([[[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],   #  red,   green,     blue
                [[  0, 255, 255], [255,   0, 255], [255, 255,   0]],   # cyan, magenta,   yellow
                [[  0,   0,   0], [255, 255, 255], [127, 127, 127]]])  #black,   white, gray 50%
plt.axis("off")
plt.imshow(img)

# Let's change the lower left pixel from Black to Orange

img[2, 0, 0] = 255 # Assign 100% to red
img[2, 0, 1] = 200 # Assign about 80% to green

plt.imshow(img[2:, :1]) # Show pixel at row 2 and column 0

# Let's display the whole image
plt.imshow(img)
