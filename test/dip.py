import cv2

import matplotlib.pyplot as plt

demo = cv2.imread('demo.jpg')
demo_r = cv2.bitwise_not(demo)
# cv2.imshow('demo', demo_r)
plt.figure('demo')
plt.imshow(demo_r)
# plt.show()

# %%
from PIL import Image
import PIL.ImageOps

image = Image.open('demo.jpg')

inverted_image = PIL.ImageOps.invert(image)
# image.show()

plt.figure('demo')
plt.imshow(inverted_image)
plt.show()

# %%
dst = cv2.Laplacian(demo, -1)
plt.imshow(dst)
plt.show()

demo_r = demo - dst
plt.imshow(demo_r)
plt.show()

import numpy as np

a = np.array([1, 0], [0, 1])
