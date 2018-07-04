import cv2
import numpy as np


def harris(img, k=0.04, thresh=2.5):
    Ix = cv2.Sobel(img, -1, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, -1, 0, 1, ksize=3)
    Ixx = cv2.Sobel(Ix, -1, 1, 0, ksize=3)
    Ixy = cv2.Sobel(Ix, -1, 0, 1, ksize=3)
    # assume second order is continous
    Iyx = Ixy
    Iyy = cv2.Sobel(Iy, -1, 0, 1, ksize=3)

    M = np.stack([Ixx, Ixy, Iyx, Iyy], -1)
    M = cv2.GaussianBlur(M, (0, 0), 1.)
    det = M[:, :, 0] * M[:, :, 3] - M[:, :, 1] * M[:, :, 2]
    trace = M[:, :, 0] + M[:, :, 3]
    coord = np.stack(np.where(det - k * trace * trace > thresh), -1)
    for i in range(coord.shape[0]):
        cv2.circle(img, (coord[i, 1], coord[i, 0]), 3, (128, 128, 128))
    return img



if __name__ == '__main__':
    img = cv2.imread('../lenna.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.
    # cv2.imshow('before', img)
    # cv2.waitKey()

    img = harris(img)
    cv2.imshow('after', img)
    cv2.waitKey()