import cv2
import scipy.ndimage as scimg
import scipy.signal as signal
import numpy as np


def bilateral(img, sigma_s=4, sigma_i=0.025, fast_approx=True):
    sigma_i = int(sigma_i * 256)

    cv2.imshow('origin', img)
    w = np.zeros([*img.shape, 256])
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    x = x.reshape(-1)
    y = y.reshape(-1)
    w[[x, y, img[[x, y]]]] = 1
    i = np.zeros_like(w)
    i[[x, y]] = np.expand_dims(img[[x, y]], -1)

    wi = w * i
    raw_size = wi.shape
    if fast_approx:
        kernel = np.ones([sigma_s, sigma_s, sigma_i])
        kernel /= kernel.sum()
        wi = signal.convolve(wi, kernel, mode='same')
        w = signal.convolve(w, kernel, mode='same')
        wi = wi[::sigma_s, ::sigma_s, ::sigma_i]
        w = w[::sigma_s, ::sigma_s, ::sigma_i]

    conv_wi = scimg.gaussian_filter(wi, [sigma_s, sigma_s, sigma_i])
    conv_w = scimg.gaussian_filter(w, [sigma_s, sigma_s, sigma_i])
    i_final = conv_wi / (conv_w + 1e-6)
    if fast_approx:
        i_final = scimg.zoom(i_final, [raw_size[i] / w.shape[i] for i in range(3)], order=1)
    filtered = i_final[[x, y, img[[x, y]]]]
    img = filtered.reshape(img.shape).transpose().astype(np.uint8)
    return img


if __name__ == '__main__':
    img = cv2.imread('lenna.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = bilateral(img)
    cv2.imshow('after', img)
    cv2.waitKey()
