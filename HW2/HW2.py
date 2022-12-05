import math
import cv2
import numpy as np
from numpy import uint8


def relu(x):
    if x < 0:
        x = 0
    if x > 255:
        x = 255
    return x


def norm(X):
    beta = np.zeros(X.shape[0], dtype=float)
    gamma = np.ones(X.shape[0], dtype=float)
    mean = np.zeros(X.shape[0], dtype=float)
    eps = 0.000000001

    for i in range(X.shape[0]):
        mean[i] = np.mean(X[i, :])
        for j in range(X.shape[1]):
            X[i, j] = ((X[i, j] - mean[i]) / math.sqrt(np.std(X[i, :]) + eps))
            X[i, j] = gamma[i] * X[i, j] + beta[i]
    return X


def conv2d(X, kernels, B):
    U = 1
    W1, H1, D1 = X.shape
    M, D2, W_k, H_k = kernels.shape
    W2 = round((W1 - W_k) / U + 1)
    H2 = round((H1 - H_k) / U + 1)
    result = np.zeros((W2, H2, M))

    for m in range(M):
        for w in range(W2):
            for h in range(H2):
                result[w, h, m] = B[m]
                for i in range(W_k):
                    for j in range(H_k):
                        for k in range(D2):
                            result[w, h, m] += X[U * w + i, U * h + j, k] * kernels[m, k, i, j]
                result[w, h, m] = relu(result[w, h, m])
    return np.asarray(result, dtype=uint8)


def pooling(X):
    U = 2
    W1, H1, C1 = X.shape
    W2 = round((W1 - 2 + 1) / U + 1)
    H2 = round((H1 - 2 + 1) / U + 1)
    result = np.zeros((W2, H2, C1), dtype=float)

    for k in range(C1):
        x = 0
        for w in np.arange(0, W1 - 2 - 1, U):
            y = 0
            for h in np.arange(0, H1 - 2 - 1, U):
                result[x, y, k] = np.max(X[w:w + 2, h:h + 2, k])
                y += 1
            x += 1
    return np.asarray(result, dtype=uint8)


def softmax(X):
    X = np.float64(X)
    return np.exp(X) / np.sum(np.exp(X))


image = cv2.imread("image.jpg")
B = np.random.rand(5)
k = np.random.rand(3, 3)
kernels = np.array([[k, k, k],
                    [k, k, k],
                    [k, k, k],
                    [k, k, k],
                    [k, k, k]])
conv_fmap = conv2d(image, kernels, B)
norm_fmap = norm(conv_fmap)
pool_fmap = pooling(norm_fmap)
probs = softmax(pool_fmap)
print(probs.shape)
