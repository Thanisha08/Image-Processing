"""
.. module:: Concatenate Different images of Mandarin Monkey
   :platform: Windows
   :synopsis: This module does simple details

.. module author:: Thanisha
.. copyrights: Karomi Technology Private Limited
.. date created: 08/08/2018

"""

import cv2
import numpy as np

image = cv2.imread('E:/mandarin_monkey.png')
image = cv2.resize(image, (0, 0), None, .25, .25)
blue_img = np.zeros(image.shape, dtype=np.uint8)
green_img = np.zeros(image.shape, dtype=np.uint8)
red_img = np.zeros(image.shape, dtype=np.uint8)
blue_img[:, :, 0] = image[:, :, 0]
green_img[:, :, 1] = image[:, :, 1]
red_img[:, :, 2] = image[:, :, 2]

numpy_horizontal1 = np.hstack((image, red_img))
numpy_horizontal2 = np.hstack((green_img, blue_img))
numpy_image = np.vstack((numpy_horizontal1, numpy_horizontal2))

cv2.imshow('Mandarin Monkey', numpy_image)
cv2.waitKey(0)
