import numpy as np
import matplotlib.pyplot as plt

# %config InlineBackend.figure_formats = 'retina'

img = np.array(
    [
        [0, 255, 0],  #   black,  white,     black
        [50, 200, 50],  #    dark,  light,      dark
        [110, 127, 140],
    ]
)  # mid-dark,   mid, mid-light

plt.imshow(img, cmap="gray")

# This is how it looks in a text representation

print(img)
type(img[0, 0])

# Now let's create a simple 3x3 RGB image.
# This time each pixel is an [R,G,B] triad.

img = np.array(
    [
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  #   red,   green,     blue
        [[0, 255, 255], [255, 0, 255], [255, 255, 0]],  #  cyan, magenta,   yellow
        [[0, 0, 0], [255, 255, 255], [127, 127, 127]],
    ]
)  # black,   white, gray 50%
plt.axis("off")
plt.imshow(img)

# This is how it looks in a text representation

print(img)
type(img[0, 0, 0])

# Now let's create the same image with floating point numbers.

img = np.array(
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  #  red,   green,     blue
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],  # cyan, magenta,   yellow
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
    ]
)  # black,   white, gray 50%
plt.imshow(img)

# This is how it looks in a text representation

print(img)
type(img[0, 0, 0])
