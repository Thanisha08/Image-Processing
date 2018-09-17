import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim



def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    cbcr[:, :, 0] = .299 * r + .587 * g + .114 * b
    cbcr[:, :, 1] = 128 - .169 * r - .331 * g + .5 * b
    cbcr[:, :, 2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)


def ycbcr2rgb(im):
    rgb = np.empty_like(im)
    im = im.astype(np.float)
    y = im[:, :, 0]
    cb = im[:, :, 1]-128
    cr = im[:, :, 2]-128
    rgb[:, :, 0] = y + 1.402 * cr
    rgb[:, :, 1] = y - .34414 * cb - .71414 * cr
    rgb[:, :, 2] = y + 1.772 * cb
    return np.uint8(rgb)



def mse(imageA, imageB):
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

img = cv2.imread('E:/mandarin_monkey.png')
imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
cv2.imshow('COLOR_RGB2YCrCb', imgYCC)
imgRGB = cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2BGR)
cv2.imshow('COLOR_YCrCb2RGB', imgRGB)
imgYCC2=rgb2ycbcr(img)
cv2.imshow('rgb2ycbcr', imgYCC2)
imgRGB2=ycbcr2rgb(imgYCC2)
cv2.imshow('ycbcr2rgb', imgRGB2)
m1 = mse(img, imgRGB)
m2 = mse(img, imgRGB2)
print("mse(img, imgRGB): %s" % (m1))
print("mse(img, imgRGB2): %s" % (m2))
cv2.waitKey(0)
cv2.destroyAllWindows()
