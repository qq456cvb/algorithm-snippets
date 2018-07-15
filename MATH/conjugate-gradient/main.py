import numpy as np

if __name__ == '__main__':
    A = np.random.rand(9, 9)
    A = (A + A.transpose()) / 2
    b = np.random.rand(9)
    x_true = np.matmul(np.linalg.inv(A), b)

    x = np.zeros(9)
    r = b
    p = r

    for i in range(9):
        alpha = r.dot(r) / p.dot(A @ p)
        x = x + alpha * p
        r_old = r
        r = r_old - alpha * (A @ p)
        beta = r.dot(r) / r_old.dot(r_old)
        p = r + beta * p
        print(np.linalg.norm(r))

