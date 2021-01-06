import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm
from sklearn.linear_model import Ridge

def findMinL1(funObj, w, L1_lambda, maxEvals, *args, verbose=0):
    """
    Uses the L1 proximal gradient descent to optimize the objective function

    The line search algorithm divides the step size by 2 until
    it find the step size that results in a decrease of the L1 regularized
    objective function
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    funEvals = 1

    alpha = 1.
    proxL1 = lambda w, alpha: np.sign(w) * np.maximum(abs(w)- L1_lambda*alpha,0)
    L1Term = lambda w: L1_lambda * np.sum(np.abs(w))

    while True:
        gtd = None
        # Start line search to determine alpha
        while True:
            w_new = w - alpha * g
            w_new = proxL1(w_new, alpha)

            if gtd is None:
                gtd = g.T.dot(w_new - w)

            f_new, g_new = funObj(w_new, *args)
            funEvals += 1

            if f_new + L1Term(w_new) <= f + L1Term(w) + gamma*alpha*gtd:
                # Wolfe condition satisfied, end the line search
                break

            if verbose > 1:
                print("Backtracking... f_new: %.3f, f: %.3f" % (f_new, f))

            # Update alpha
            alpha /= 2.

        # Print progress
        if verbose > 0:
            print("%d - alpha: %.3f - loss: %.3f" % (funEvals, alpha, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(w - proxL1(w - g, 1.0), float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break
        print(funEvals)
    return w, f


class logRegL1:
    # Logistic Regression
    def __init__(self, L1_constant=1, verbose=0, maxEvals=1000):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.L1_constant = L1_constant

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        # utils.check_gradient(self, X, y)
        (self.w, f) = findMinL1(self.funObj, self.w, self.L1_constant,
                                self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return X @ self.w


def processX(trainsize=100):
    newdf = np.empty((0, 22))
    for i in range(trainsize):
        cs1 = pd.read_csv("train/X/X_" + str(i) + ".csv")
        cs2 = cs1.eq(' agent', axis=1).any(axis=0)
        agentpos = cs2[cs2].index[0][-1]
        cs2 = cs1[[' x' + agentpos, ' y' + agentpos]]
        cs2 = np.array(cs2.values.flatten())
        newdf = np.append(newdf, [cs2], axis=0)

    return newdf


def processy(trainsize=100):
    newdf = np.empty((0, 60))
    for i in range(trainsize):
        cs1 = pd.read_csv("train/y/y_" + str(i) + ".csv", usecols=[' x', ' y'])
        cs2 = np.array(cs1.values.flatten())
        cs2len = len(cs2)
        cs2last = cs2[-1]
        for j in range(60 - cs2len):
            cs2 = np.append(cs2, cs2last)
        newdf = np.append(newdf, [cs2], axis=0)

    return newdf


def valXprocess(testsize=100, flag=0):
    if flag == 0:
        newdf = np.empty((0, 22))
        for i in range(testsize):
            cs1 = pd.read_csv("val/X/X_" + str(i) + ".csv")
            cs2 = cs1.eq(' agent', axis=1).any(axis=0)
            agentpos = cs2[cs2].index[0][-1]
            cs2 = cs1[[' x' + agentpos, ' y' + agentpos]]
            cs2 = np.array(cs2.values.flatten())
            newdf = np.append(newdf, [cs2], axis=0)
    else:
        newdf = np.empty((0, 22))
        for i in range(testsize):
            cs1 = pd.read_csv("test/X/X_" + str(i) + ".csv")
            cs2 = cs1.eq(' agent', axis=1).any(axis=0)
            agentpos = cs2[cs2].index[0][-1]
            cs2 = cs1[[' x' + agentpos, ' y' + agentpos]]
            cs2 = np.array(cs2.values.flatten())
            newdf = np.append(newdf, [cs2], axis=0)

    return newdf

def valyprocess(testsize=100):
    newdf = np.empty((0, 60))
    for i in range(testsize):
        cs1 = pd.read_csv("val/y/y_" + str(i) + ".csv", usecols=[' x', ' y'])
        cs2 = np.array(cs1.values.flatten())
        cs2len = len(cs2)
        cs2last = cs2[-1]
        for j in range(60 - cs2len):
            cs2 = np.append(cs2, cs2last)
        newdf = np.append(newdf, [cs2], axis=0)

    return newdf

def threesecpred(trainsize=1007, valshape=19, flag=0):
    # valshape += 1
    pred = np.zeros((valshape, 60))
    for i in range(30):
        X = processX(trainsize)
        y = processy(trainsize)
        #hidden_layer_sizes = [10]
        # regx = NeuralNet(hidden_layer_sizes)
        # regx=LinearRegression()
        regx = logRegL1()
        regx.fit(X, y[:, 2 * i])
        valx = valXprocess(testsize=valshape, flag=flag)
        # valy=valyprocess(testsize=valshape)
        pred[:, 2 * i] = regx.predict(valx).flatten()

        # print(pred)
        # print("score"+str(2*i),regx.score(valx,valy[:,2*i]))
        # regy=LinearRegression()
        # regy = NeuralNet(hidden_layer_sizes)
        regy = logRegL1()
        regy.fit(X, y[:, 2 * i + 1])
        pred[:, 2 * i + 1] = regy.predict(valx).flatten()
        # print("score"+str(2*i+1),regy.score(valx,valy[:,2*i+1]))
        # print(pred)
    return pred


#newpred=threesecpred(trainsize=2308,valshape=524,flag=0);

#valy=valyprocess(testsize=524)
#rsme=np.mean((valy-newpred)**2)

#print("rsme",rsme)

newpred2=threesecpred(trainsize=2308,valshape=20,flag=1)
newpred2=newpred2.flatten()
np.savetxt("submission_logreg.csv",newpred2,delimiter=",")