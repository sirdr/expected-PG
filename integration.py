import scipy.integrate as integrate
import numpy as np


def f(*args):
    return np.sum(args)



def compute_integral_over_action_space(f, limits, tol=None):
    result = integrate.nquad(f,limits)
    return result[0]

action= np.array([1,2, 3, 4])
limits = [[-5,1], [-9,1], [0,10], [-3,3]]

result = compute_integral_over_action_space(f,limits)
print(result)


#def f(*args):
#    print(args[0][1])
#    return np.sum(args)

#print(f([0,1,2]))
