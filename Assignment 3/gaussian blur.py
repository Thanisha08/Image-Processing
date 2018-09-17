import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

img = cv2.imread('E:/mandarin_monkey.png',0) # Load the image
kernel = np.ones((20, 20),dtype="float")
edges = scipy.signal.convolve2d(img, kernel, 'valid')
plt.imshow(edges,cmap=plt.cm.gray)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
