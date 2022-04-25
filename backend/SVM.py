import sys
import pandas
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def SVM_DUAL(K, loss, C, eps, y, maxiter):
    if loss == 'hinge':
        pass
    elif loss == 'quadratic':
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                K[i, j] = K[i, j] + 1/(2 * C) * delta[i, j]
    K = K + 1
    n = K.shape[0]
    etas = np.zeros(n)
    for k in range(n):
        etas[k] = 1/K[k, k]
    t = 0
    a = np.zeros([n, 1])
    als = []
    als.append(a)
    while True:
        a = als[t]
        perm = np.random.permutation(n)
        for i in range(n):
            tmp = 0
            k = perm[i] - 1
            for i in range(n):
                tmp += a[i, 0] * y[i, 0] * K[i, k]
            a[k] = a[k] + etas[k] * (1 - y[k] * tmp)
            if a[k] < 0:
                a[k] = 0
            if loss == 'hinge' and a[k] > C:
                a[k] = C
        als.append(a)
        t = t + 1
        if np.linalg.norm(als[t] - als[t - 1]) <= eps or t >= maxiter:
            break
    return als[-1]

def Cal_w_b(a, C, d, X, y):
    w = np.zeros([1, d]) 
    cnt = 0
    for i in range(a.shape[0]):
        if a[i, 0] > 0 and a[i, 0] < C:
            w += a[i, 0] * y[i, 0] * X[i, :]
            cnt += 1
            print(X[i, :])
    b = 0
    for i in range(a.shape[0]):
        if a[i, 0] > 0 and a[i, 0] < C:
            b += y[i, 0] - w.dot(X[i, :])
    b = b/cnt
    return w, b, cnt

def Cal_w_b_new(a, C, d, X, y):
    cnt = 0
    for i in range(a.shape[0]):
        if a[i, 0] > 0 and a[i, 0] < C:
            cnt += 1
    return cnt

def Testing(a, X, z, y, C):
    ans = 0
    for i in range(a.shape[0]):
        if a[i, :]>0 and a[i,:]<C:
            ans += a[i, :] * y[i, :] * (X[i, :].dot(z) + 1)
    if ans > 0:
        return 1
    else:
        return -1

def main():    
    address = sys.argv[1]
    C = float(sys.argv[2])
    eps = float(sys.argv[3])
    maxiter = int(sys.argv[4])
    kernel = sys.argv[5]
    kernel_param = float(sys.argv[6])
    X = pandas.read_csv(open(address), delimiter=",")
    del X['date']
    del X['rv2']
    n = X.shape[0]
    d = X.shape[1]
    X = X.to_numpy()
    y = X[:, 0]
    y_feature = y.copy()
    y_feature[y > 50] = -1
    y_feature[y <= 50] = 1
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = X[:, 1:]
    training = X[0: 5000]
    training_y = y_feature[0: 5000]
    training_y = training_y.reshape(training_y.shape[0], 1)
    validation = X[5000: 7000]
    validation_y = y_feature[5000: 7000]
    validation_y = validation_y.reshape(validation_y.shape[0], 1)
    testing = X[7000: 12000]
    testing_y = y_feature[7000: 12000]
    testing_y = testing_y.reshape(testing_y.shape[0], 1)

    d_train = 5000
    K = np.zeros([d_train, d_train])
    if kernel == 'linear':
        for i in range(d_train):
            for j in range(d_train):
                K[i, j] = training[i, :].dot(training[j, :])
    elif kernel == 'gaussian':
        for i in range(d_train):
            for j in range(d_train):
                tmp = training[i, :]-training[j, :]
                K[i, j] = math.exp(-sum(tmp * tmp)/(2 * kernel_param))
    elif kernel == 'polynomial':
        for i in range(d_train):
            for j in range(d_train):
                K[i, j] = math.pow(training[i, :].dot(training[j, :]) + 1, kernel_param)
    a = SVM_DUAL(K, 'hinge', C, eps, training_y, maxiter)
    print(a)
    #w, b, cnt = Cal_w_b(a, C, training.shape[1], training, training_y)
    cnt = Cal_w_b_new(a, C, training.shape[1], training, training_y)
    print("cnt: ", cnt)
    ans = 0
    for i in range(testing.shape[0]):
        y_hat = Testing(a, testing, testing[i, :], testing_y, C)
        if y_hat == testing_y[i]:
            ans +=1
    print("Acc: ", ans/testing.shape[0])
    

if __name__ == '__main__':
    main()