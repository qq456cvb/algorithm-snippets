import numpy as np
import cv2


h = 0.
beta = 1.
eta = 2.1
noise_level = 0.1

def energy(x, y):
    return h * np.sum(x) - beta * (np.sum(x[:, :1] * x[:, 1:]) + np.sum(x[:1, :] * x[1:, :])) - eta * np.sum(x * y)


if __name__ == "__main__":
    img = cv2.imread('./CV/lenna.jpg')
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    y = np.clip(img - (np.random.rand(*img.shape) < noise_level) * 255., 0, 255) / 255.

    # normalize to [-1, 1]
    y = y * 2. - 1.
    x = y.copy()

    min_energy = energy(x, y)
    for _ in range(1000000):
        i = np.random.randint(1, x.shape[0] - 1)
        j = np.random.randint(1, x.shape[1] - 1)
        cand = -x[i, j]
        
        cost = -beta * cand * (x[i + 1, j] + x[i - 1, j] + x[i, j + 1] + x[i, j - 1]) - eta * cand * y[i, j]
        if cost < 0:
            min_energy += cost
            x[i, j] = cand
            print('min energy', min_energy)
    x = (x + 1) / 2
    y = (y + 1) / 2
    cv2.imshow('result', x * 255.)
    cv2.imshow('origin', y * 255.)
    cv2.waitKey()
    