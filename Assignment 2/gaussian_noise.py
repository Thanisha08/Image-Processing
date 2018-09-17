
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('E:/mandarin_monkey.png', 1)
row, col, ch = image.shape
mean = 0.0
var = 40
sigma = var * 0.5
gauss = np.random.normal(mean, sigma, (row, col, ch))
gauss = gauss.reshape(row, col, ch)
gauss_img = image + gauss
gauss_img=gauss_img.astype(np.uint8)
plt.imshow(cv2.cvtColor(gauss_img, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Noise with mean=0 and Variance=40'), plt.xticks([]), plt.yticks([])
plt.show()
