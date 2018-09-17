import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('E:\smoke.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# calculate amplitude spectrum
mag_spec = 20*np.log(np.abs(fshift))
r = f.shape[0]/2        # number of rows/2
c = f.shape[1]/2        # number of columns/2
r=int(r)
c=int(c)
p = 3
n = 1
fshift2 = np.copy(fshift)
fshift2[253:259, 0:255]= 0.000001
fshift2[253:259, 257:512]=0.000001
mag_spec2 = 20*np.log(np.abs(fshift2))
inv_fshift = np.fft.ifftshift(fshift2)
img_recon = np.real(np.fft.ifft2(inv_fshift))


plt.subplot(331),plt.imshow(np.log(np.abs(f.real)), cmap = 'gray')
plt.title('fft_real'), plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(np.log(np.abs(f.imag)), cmap = 'gray')
plt.title('fft_imaginary'), plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(np.log(np.abs(f)), cmap = 'gray')
plt.title('fft'), plt.xticks([]), plt.yticks([])

plt.subplot(334),plt.imshow(np.log(np.abs(fshift.real)), cmap = 'gray')
plt.title('fft_fshift_real'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(np.log(np.abs(fshift.imag)), cmap = 'gray')
plt.title('fft_shift)imaginary'), plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(np.log(np.abs(fshift)), cmap = 'gray')
plt.title('fft_fshift'), plt.xticks([]), plt.yticks([])

plt.subplot(337),plt.imshow(np.log(np.abs(inv_fshift)), cmap = 'gray')
plt.title('fft_ishift_real'), plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(np.log(np.abs(inv_fshift.real)), cmap = 'gray')
plt.title('fft_ishift_imaginary'), plt.xticks([]), plt.yticks([])
plt.subplot(339),plt.imshow(np.log(np.abs(inv_fshift.imag)), cmap = 'gray')
plt.title('fft_ishift'), plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(221),plt.imshow(mag_spec, cmap = 'gray')
plt.title('Input Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(mag_spec2, cmap = 'gray')
plt.title('Modified Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img_recon, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()
