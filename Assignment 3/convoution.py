import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
#from skimage import io, color
#from skimage import exposure

img = cv2.imread('E:/mandarin_monkey.png',0) # Load the image
#img = color.rgb2gray(img1)
kernel = np.array([[0,0,0],[0,-1,1],[0,0,0]])
kernel2 = np.array([[0,1,0],[0,-1,0],[0,0,0]])
kernel45 = np.array([[0,0,1],[0,-1,0],[0,0,0]])
kernel20 = np.array([[1,0,0],[0,-1,0],[0,0,0]])
edges = scipy.signal.convolve2d(img, kernel, 'valid')
edges2 = scipy.signal.convolve2d(img, kernel2, 'valid')
edges45 = scipy.signal.convolve2d(img, kernel45, 'valid')
edges120 = scipy.signal.convolve2d(img, kernel20, 'valid')
f = plt.figure()
f.add_subplot(1,4, 1)
plt.title('0 Degree')
plt.imshow(edges,cmap=plt.cm.gray)
f.add_subplot(1,4, 2)
plt.title('90 Degree')
plt.imshow(edges2,cmap=plt.cm.gray)
f.add_subplot(1,4, 3)
plt.title("45 Degree")
plt.imshow(edges45,cmap="gray")
f.add_subplot(1,4, 4)
plt.title('120 Degree')
plt.imshow(edges120,cmap=plt.cm.gray)
plt.show(block=True)
cv2.waitKey(0)
cv2.destroyAllWindows()
