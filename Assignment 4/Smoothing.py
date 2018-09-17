import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

img = cv2.imread('E:\smoke.png',0)
kernel = np.ones((15, 15),dtype="float")/225
edges = scipy.signal.convolve2d(img, kernel, 'valid')
plt.imshow(edges,cmap=plt.cm.gray)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
