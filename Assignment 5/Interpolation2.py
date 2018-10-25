import cv2
import numpy as np


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


img = cv2.imread('E:/mandarin_monkey.png')
cv2.imwrite('E:/monkey_jpg.jpg',img)
cv2.imwrite('E:/monkey_png.png',img)
img1 = cv2.imread('E:/monkey_jpg.jpg')
img2 = cv2.imread('E:/monkey_png.png')
m1 = mse(img, img1)
m2 = mse(img, img2)
print(m1)
print(m1)
diff1=cv2.absdiff(img,img1)
diff2=cv2.absdiff(img,img2)
print(diff1)
print(diff2)
cv2.imshow("diff1-JPG",diff1)
cv2.imshow("diff2-PNG",diff2)
cv2.waitKey(0)

