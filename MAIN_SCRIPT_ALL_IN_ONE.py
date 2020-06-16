#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sklearn
import pandas as pd
#from SpectrumKernel import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import random

#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# ## Data Loading

X = pd.read_csv('Data/Xtr.csv')
X_test = pd.read_csv('Data/Xte.csv')
y = pd.read_csv('Data/Ytr.csv')


data = pd.DataFrame()
X['type'] = 'train' 
X_test['type'] = 'test'


data = pd.concat([X, X_test])


# ## Data preprocessing


def kmers(sequence, size=None):
    return [sequence[x : x + size].lower() for x in range(len(sequence) - size + 1)]


def get_kmers(data, k):
    seq = []
    for i in range(len(data)):
        sequence = data.iloc[i]['seq']
        sequence = kmers(sequence, k)
        seq.append(sequence)
    #names = ['txt'+str(i) for i in range(len(word))]
    return seq


data_all = np.array(get_kmers(data, k = 3))
data_all = pd.DataFrame(data_all)


def getOnehotEncoding(X):
    enc = OneHotEncoder(sparse=False)
    enc.fit(X)
    features = enc.transform(X)
    return features


data_all = getOnehotEncoding(data_all)


data_train = data_all[0:len(X), :]
data_test = data_all[len(X):, :]


# In[14]:


y_dna = y.Bound.values
y_dna = np.where(y_dna == 0, -1, 1)

import cvxopt

def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    #cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
    return np.array(solution['x']).flatten()

solve_qp = cvxopt_qp


# ## Kernels

def rbf_kernel(X1, X2, sigma=10):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the RBF kernel with parameter sigma
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    sigma: float
    '''
    
    X2_norm = np.sum(X2 ** 2, axis = -1)
    X1_norm = np.sum(X1 ** 2, axis = -1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K

def sigma_from_median(X):
    '''
    Returns the median of ||Xi-Xj||
    
    Input
    -----
    X: (n, p) matrix
    '''
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)


# In[18]:


def linear_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the linear kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return X1.dot(X2.T)

def quadratic_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the quadratic kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return (1 + linear_kernel(X1, X2))**2  


# ## Kernel Methods

class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    
    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'linear': linear_kernel,
        'quadratic': quadratic_kernel,
        'rbf': rbf_kernel,
    }
    def __init__(self, kernel='linear', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf':
            params['sigma'] = kwargs.get('sigma', None)
        return params

    def fit(self, X, y, **kwargs):
        return self
        
    def decision_function(self, X):
        pass

    def predict(self, X):
        pass


# ## Kernel Logistic Regression

class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd = 0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit(self, X, y, sample_weights=None):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        if sample_weights is not None:
            w_sqrt = np.sqrt(sample_weights)
            self.X_train = self.X_train * w_sqrt[:, None]
            self.y_train = self.y_train * w_sqrt
            
        A = self.kernel_function_(X, X, **self.kernel_parameters)
        A[np.diag_indices_from(A)] += n * lambd
        
        # self.alpha = (K + n lambda I)^-1 y
        self.alpha = np.linalg.solve(A , self.y_train)

        return self
    
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        #K_x = (K_x - K_x.min()) /(K_x.max() - K_x.min())
        return K_x.dot(self.alpha)
    
    def predict(self, X):
        return self.decision_function(X)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class KernelLogisticRegression(KernelMethodBase):
    '''
    Kernel Logistic Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd

        super(KernelLogisticRegression, self).__init__(**kwargs)

    def fit(self, X, y, max_iter=100, tol=1e-5):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        K = self.kernel_function_(X, X, **self.kernel_parameters)
        #K = (K - K.min()) /(K.max() - K.min())
        
        # IRLS
        KRR = KernelRidgeRegression(
            lambd=2*self.lambd,
            kernel=self.kernel_name,
            **self.kernel_parameters
        )
        # Initialize
        alpha = np.zeros(n)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            m = K.dot(alpha_old)
            w = sigmoid(m) * sigmoid(-m)
            z = m + self.y_train / sigmoid(self.y_train * m)
            alpha = KRR.fit(self.X_train, z, sample_weights = w).alpha
            # Break condition (achieved convergence)
            if np.sum((alpha - alpha_old)**2) < tol:
                break

        self.n_iter = n_iter
        self.alpha = alpha

        return self
            
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        #K_x = (K_x - K_x.min()) /(K_x.max() - K_x.min())
        return sigmoid(K_x.dot(self.alpha))

    def predict(self, X):
        proba = self.decision_function(X)
        predicted_classes = np.where(proba < 0.5, -1, 1)
        return predicted_classes


# Prediction error
def get_error(ypred, ytrue):
    e = (ypred != ytrue).mean()
    return e


def svm_dual_soft_to_qp_kernel(K, y, C=1):
    n = K.shape[0]
    assert (len(y) == n)
        
    # Dual formulation, soft margin
    K = (K - K.min()) /(K.max() - K.min())
    P = np.diag(y).dot(K).dot(np.diag(y))
    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    P += eps * np.eye(n)
    #print(f'P: {P.shape}')
    q = - np.ones(n)
    #print(f'q: {q.shape}')
    G = np.vstack([-np.eye(n), np.eye(n)])
    #print(f'G: {G.shape}')
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    #print(f'h: {h.shape}')
    A = y[np.newaxis, :]
    A = A.astype('float')
    #print(f'A.typecode: {A.typecode}')
    #print(f'A: {A.shape}')
    b = np.array([0.])
    return P, q, G, h, A, b


class KernelSVM(KernelMethodBase):
    '''
    Kernel SVM Classification
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, C=0.1, **kwargs):
        self.C = C
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelSVM, self).__init__(**kwargs)

    def fit(self, X, y, tol=1e-3):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        # Kernel matrix
        K = self.kernel_function_(X, X, **self.kernel_parameters)
        K = (K - K.min()) /(K.max() - K.min())
        #print(f'K train: {K.shape}')
        
        # Solve dual problem
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.C))
        
        # Compute support vectors and bias b
        sv = np.logical_and((self.alpha > tol), (self.C - self.alpha > tol))
        self.bias = y[sv] - K[sv].dot(self.alpha * y)
        self.bias = self.bias.mean()

        self.support_vector_indices = np.nonzero(sv)[0]

        return self
        
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        K_x = (K_x - K_x.min()) /(K_x.max() - K_x.min())
        #print(f'K_x decision: {K.shape}')
        return K_x.dot(self.alpha * self.y_train) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X))


def crossVal(X, y, k, kernel, C, sigma, lambd, model, flag=True):
    kf = KFold(n_splits= k, shuffle=True, random_state=42)
    error_tracking = []
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        if model == 'svm':
            model = KernelSVM(C=C, kernel=kernel, sigma=sigma)
        if model == 'log':
            model = KernelLogisticRegression(lambd=lambd, kernel=kernel, sigma=sigma)

        y_p = model.fit(X_train, y_train).predict(X_val)
        er = np.mean(y_p != y_val)
      
        error_tracking.append(er)
        if flag:
            print(f'Fold {i} ---> Test error: {er * 100} %')
    return np.mean(error_tracking)


if __name__== '__main__':

	kernel = 'rbf'
	# {'C': 66, 'sigma': 5} --> 66.39
	C = 66
	sigma = 5

        # Cross-validation
	print('Cross-validation ......\n')
	average_error = crossVal(data_train, y_dna, 5, kernel='rbf', C=66, sigma=5, lambd=1, model='svm', flag=True)
	print('Average Test Error: {} %\n'.format(average_error * 100))

        # Inference
	print('Inference .....\n')
	model = KernelSVM(C=C, kernel=kernel, sigma=sigma)
	predicted_labels = model.fit(data_train, y_dna).predict(data_test)



	print(f'number of 1 predicted: {np.sum(predicted_labels == 1.)}, number of -1 predicted: {np.sum(predicted_labels == -1.)}\n')


	labels = np.where(predicted_labels < 0, 0, 1)


	sub_file = pd.DataFrame()
	sub_file['Id'] = X_test.Id.tolist()
	sub_file['Bound'] = list(labels)

	#print(sub_file.head())
	print('Submission file saved!')

	sub_file.to_csv('submission_file.csv', index=False)



