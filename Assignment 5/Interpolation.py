import cv2
import numpy as np


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


img = cv2.imread('E:/mandarin_monkey.png')
r = 100.0 / img.shape[1]
dim = (int(img.shape[1] * r),int(img.shape[0] * r))
dim2 = (837,823)
linear =  cv2.resize(img,dim,interpolation=cv2.INTER_LINEAR)
cubic =  cv2.resize(img,dim,interpolation=cv2.INTER_CUBIC)
area = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
cv2.imwrite("resizeimg.png",linear)
cv2.imwrite("resizeimg2.png",cubic)
cv2.imwrite("resizeimg3.png",area)
cv2.imshow("Downsample Linear",linear)
cv2.imshow("Downsample Cubic",cubic)
cv2.imshow("Downsample Area",area)
img1 = cv2.imread('E:/resizeimg.png')
img2 = cv2.imread('E:/resizeimg2.png')
img3 = cv2.imread('E:/resizeimg3.png')
linear_inverse =  cv2.resize(img1,dim2,interpolation=cv2.INTER_LINEAR)
cubic_inverse =  cv2.resize(img2,dim2,interpolation=cv2.INTER_CUBIC)
area_inverse = cv2.resize(img3,dim2,interpolation=cv2.INTER_AREA)
cv2.imshow("Upsample Linear",linear_inverse)
cv2.imshow("Upsample Cubic",cubic_inverse)
cv2.imshow("Upsample Area",area_inverse)

m1 = mse(img, linear_inverse)
m2 = mse(img, cubic_inverse)
m3 = mse(img, area_inverse)
print(m1)
print(m1)
print(m2)

diff1=cv2.absdiff(img,linear_inverse)
diff2=cv2.absdiff(img,cubic_inverse)
diff3=cv2.absdiff(img,area_inverse)
print(diff1)
print(diff2)
print(diff3)
cv2.waitKey(0)
