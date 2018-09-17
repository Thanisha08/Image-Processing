"""
.. module:: Brighten and Darken images
   :platform: Windows
   :synopsis: This module does simple details

.. module author:: Thanisha
.. copyrights: Karomi Technology Private Limited
.. date created: 08/08/2018

"""

import cv2
import numpy as np


def adjust_gamma(image, gamma2=1.0):
    """
    this function brightens or darken given images based on gamma value
    :param d, e: image to be modified
    :return: returns modified image
    """
    in_gamma = 1.0 / gamma2
    table = np.array([((i / 255.0) ** in_gamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


original = cv2.imread('E:/mandarin_monkey.png')
gamma = -3
adjusted = adjust_gamma(original, gamma2=gamma)
cv2.imshow("Modified image", adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
