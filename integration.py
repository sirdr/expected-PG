import scipy.integrate as integrate
import numpy as np


#def f(*args):
#    v = np.zeros(3)
#    for i in range(0,3):
#        v[i] = np.sum(args)
#    return v


#def compute_integral_over_action_space(f, low, high, tol=None):
#    limits = []
#    for i in range(0,len(low)):
#        lim = [low[i], high[i]]
#        limits.append(lim)

#    result = integrate.nquad(f,limits)
#    return result[0]


#low = [-5, -9, 0 ,-3]
#high = [1, 1, 10, 3]
#result = compute_integral_over_action_space(f,low, high)
#print(result)


#def f(a):
#    mat = np.zeros((3,3))
#    for i in range(0,3):
#        mat[0][i] = a+1
#    for i in range(0,3):
#        mat[1][i] = a+2
#    for i in range(0,3):
#        mat[2][i]= a+3
#    return mat


def compute_integral(f, low, high, shape, prec):
    integral = np.zeros(shape)
    a_range = np.linspace(low, high, (high-low)/prec)
    for a in a_range:
        integral += f(a)*prec
    return integral

#result = compute_integral(f, 0, 2, (3,3), .001)
#print(result)
