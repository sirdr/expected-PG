import scipy.integrate as integrate
import numpy as np


def f(x0,x1,x2,x3):
    return x0+x1+x2+x3


def compute_integral_over_action_space(f, limits, tol=None):
    result = integrate.nquad(f,limits)
    return result[0]

#action= np.array([1,2])
limits = [[-5,1], [-9,1], [0,10], [-3,3]]

result = compute_integral_over_action_space(f,limits)
print(result)
