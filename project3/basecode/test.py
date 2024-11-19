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

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    
    # calculate mean
    d = X.shape[1]
    labels = y.reshape(y.size)
    unique_labels = np.unique(labels)
    unique_label_size = unique_labels.shape[0]
    means = np.zeros((d, np.unique(labels).shape[0]))

    for i in range(unique_label_size):
        means[:, i] = np.mean(X[labels == unique_labels[i]], axis=0)

    covmat = np.cov(X, rowvar=False)

    return means, covmat


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    # Ensure inputs are numpy arrays
    y = np.array(y).flatten()  # Flatten y to a 1D array

    # Number of features and unique classes
    N, d = X.shape  # Number of training examples and feature dimension
    unique_classes = np.unique(y)  # Unique class labels
    k = len(unique_classes)  # Number of classes

    # Initialize outputs
    means = np.zeros((d, k))  # d x k matrix for means
    covmats = []  # List of k covariance matrices

    # Compute means and covariance matrices for each class
    for i, label in enumerate(unique_classes):
        X_cls = X[y == label]
        mean_cls = X_cls.mean(axis=0)
        covmat_cls = np.cov(X_cls, rowvar=False)

        means[:, i] = mean_cls  # Store mean as a column
        covmats.append(covmat_cls)  # Append covariance matrix

    return means, covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ytest = np.array(ytest).flatten()  # Flatten ytest to a 1D array

    N, d = Xtest.shape
    k = means.shape[1]

    # Inverse of the shared covariance matrix
    covmat_inv = np.linalg.inv(covmat)

    results = np.zeros((N, k))

    for i in range(k):
        mean_k = means[:, i]
        for j in range(N):
            x = Xtest[j]
            results[j, i] = x @ covmat_inv @ mean_k - 0.5 * (mean_k.T @ covmat_inv @ mean_k)

    ypred = np.argmax(results, axis=1) + 1
    ypred = ypred.reshape(-1, 1)

    acc = np.mean(ypred.flatten() == ytest)

    return acc, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ytest = np.array(ytest).flatten()

    N, d = Xtest.shape
    K = means.shape[1]

    results = np.zeros((N, K))

    for k in range(K):
        mean_k = means[:, k]
        covmat_k = covmats[k]
        covmat_k_inv = np.linalg.inv(covmat_k)
        covmat_k_det = np.linalg.det(covmat_k)

        for i in range(N):
            x = Xtest[i]
            diff = x - mean_k
            results[i, k] = (
                    -0.5 * diff.T @ covmat_k_inv @ diff
                    - 0.5 * np.log(covmat_k_det)
                    + np.log(1 / K)  # Assuming equal priors for simplicity
            )

    ypred = np.argmax(results, axis=1) + 1
    ypred = ypred.reshape(-1, 1)

    # Calculate the accuracy
    acc = np.mean(ypred.flatten() == ytest)

    return acc, ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    # w = (X^T X)^-1 X^T y
    XTX = X.T @ X  # d x d
    XTy = X.T @ y  # d x 1
    w = np.linalg.inv(XTX) @ XTy  # d x 1

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    # w=(x^T*X + λI)^(-1)X^T*y
    N, d = X.shape

    # Compute weights using the Ridge Regression formula
    I = np.eye(d)  # Identity matrix of size d x d
    XTX = X.T @ X  # d x d
    XTy = X.T @ y  # d x 1

    # Ridge regression formula: w = (X^T X + λI)^-1 X^T y
    w = np.linalg.inv(XTX + lambd * I) @ XTy  # d x 1

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
    Xp = np.zeros((N, p + 1))
    for i in range(p + 1):
        Xp[:, i] = x**i

    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
# i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Maximum number of iterations.                                                
w_init = np.ones((X_i.shape[1],1)).flatten()
optimal_lambda = 0
min_mse = float('inf')
for i, lambd in enumerate(lambdas):
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    # i = i + 1
    if mses4[i] < min_mse:
        optimal_lambda = lambd
        min_mse = mses4[i]

print(f"optimal_lambda = {optimal_lambda} for problem 4")

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
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

# Find optimal p
optimal_p_lambda_0 = np.argmin(mses5[:, 0])
optimal_p_lambda_opt = np.argmin(mses5[:, 1])
mse_lambda_0 = mses5[optimal_p_lambda_0, 0]
mse_lambda_opt = mses5[optimal_p_lambda_opt, 1]
print(f"Optimal p = {optimal_p_lambda_0} for lambda = 0 with mse = {mse_lambda_0}")
print(f"Optimal p = {optimal_p_lambda_opt} for lambda = {lambda_opt} with mse = {mse_lambda_opt}")

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
