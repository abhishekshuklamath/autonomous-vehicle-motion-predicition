import numpy as np
from numpy.linalg import norm

def findMin(funObj, w, maxEvals, *args, verbose=0):
   
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    funEvals = 1
    count=0
    alpha = 0.001
    while True:
        gg = g.T.dot(g)
        w_new = w - alpha * g
        f_new, g_new = funObj(w_new, *args)
        funEvals += 1
        if np.abs(f_new-f) <= gamma * alpha*gg:
                break

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w, f
