import cv2
from matplotlib import pyplot as plt
import numpy as np


img = cv2.imread('E:/mandarin_monkey.png',0)

# Gaussian Pyramid
layer = img.copy()
gaussian_pyramid = [layer]
for j in range(7):
    layer = cv2.pyrDown(layer)
    gaussian_pyramid.append(layer)

# Laplacian Pyramid
layer = gaussian_pyramid[5]
print(layer.shape)
#cv2.imshow("6", layer)
laplacian_pyramid = [layer]
for i in range(5, 0, -1):
    size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
    laplacian_pyramid.append(laplacian)

new1=laplacian_pyramid[0]
new1 = cv2.copyMakeBorder(new1, 0,0,78, 0, cv2.BORDER_CONSTANT)
new2=laplacian_pyramid[1]
new2 = cv2.copyMakeBorder(new2, 0,0,52, 0, cv2.BORDER_CONSTANT)
numpy_image = np.vstack((new2, new1))
print(numpy_image.shape)
new3=laplacian_pyramid[2]
numpy_image2 = np.vstack((new3, numpy_image))
new3 = cv2.copyMakeBorder(new3, 0,0,0, 0, cv2.BORDER_CONSTANT)
numpy_image2 = cv2.copyMakeBorder(numpy_image2,25 ,0,0, 0, cv2.BORDER_CONSTANT)
new4=laplacian_pyramid[3]
new4 = cv2.copyMakeBorder(new4, 0,0,0, 0, cv2.BORDER_CONSTANT)
numpy_image3 = np.hstack((new4, numpy_image2))
new5=laplacian_pyramid[4]
new5 = cv2.copyMakeBorder(new5, 0,0,0, 0, cv2.BORDER_CONSTANT)
numpy_image3 = cv2.copyMakeBorder(numpy_image3, 0,0,52, 52, cv2.BORDER_CONSTANT)
numpy_image4 = np.vstack((new5, numpy_image3))
new6=laplacian_pyramid[5]
new6 = cv2.copyMakeBorder(new6, 0,0,0,0, cv2.BORDER_CONSTANT)
numpy_image4 = cv2.copyMakeBorder(numpy_image4, 102,103,0,0, cv2.BORDER_CONSTANT)
numpy_image5 = np.hstack((new6, numpy_image4))
cv2.imshow("new5", numpy_image5)
cv2.imwrite('E:/laplacian_pyramids.png  ',numpy_image5)
#cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()