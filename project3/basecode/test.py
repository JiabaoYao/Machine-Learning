import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    # w=(x^T*X + λI)^(-1)X^T*y
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T @ X + lambd * I) @ (X.T @ y)     

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD
    predictions = Xtest @ w  # Shape (N, 1)
    mse = np.sum((ytest - predictions)**2) / ytest.shape[0]

    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD   
    # reshape weight vector into column vector
    w_T = w.reshape(-1, 1)
    # calculate residuals
    distance = y - X @ w_T

    # calculate squared error and regularization terms
    error = np.sum(distance**2) / 2 + lambd * np.sum(w_T**2) / 2
    error_grad = -X.T @ distance + lambd * w_T
    error_grad = error_grad.flatten()

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 

    # IMPLEMENT THIS METHOD
    N = x.shape[0] # Number of samples
    Xp = np.zeros(N, p + 1)
    for i in range(p + 1):
        Xp[:, i] = x**i

    return Xp

# Load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')


# Add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)


# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for i, lambd in enumerate(lambdas):
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)

# Optimal lambda
lambda_opt = lambdas[np.argmin(mses3)]

# Problem 4
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 20}
w_init = np.ones(X_i.shape[1])  # 1D array
for i, lambd in enumerate(lambdas):
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = w_l.x.reshape(-1, 1)
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)

# Plot results for Problem 3 and 4
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train, 'bo', label='Using scipy.minimize')
plt.plot(lambdas, mses3_train, label='Direct minimization')
plt.title('MSE for Train Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4, 'bo', label='Using scipy.minimize')
plt.plot(lambdas, mses3, label='Direct minimization')
plt.title('MSE for Test Data')
plt.legend()
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

# Find the optimal p for each setting
optimal_p_no_reg = np.argmin(mses5[:, 0])  # Optimal p for λ = 0
optimal_p_reg = np.argmin(mses5[:, 1])    # Optimal p for λ = lambda_opt

print(f"Optimal p for λ = 0 (no regularization): {optimal_p_no_reg}")
print(f"Optimal p for λ = {lambda_opt:.3f} (regularization): {optimal_p_reg}")

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()