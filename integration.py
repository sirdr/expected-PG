import scipy.integrate as integrate
import numpy as np


def f(*args):
    return np.sum(args)


def compute_integral_over_action_space(f, low, high, tol=None):
    limits = []
    for i in range(0,len(low)):
        lim = [low[i], high[i]]
        limits.append(lim)
    result = integrate.nquad(f,limits)
    return result[0]


#low = [-5, -9, 0 ,-3]
#high = [1, 1, 10, 3]
#result = compute_integral_over_action_space(f,low, high)
#print(result)
