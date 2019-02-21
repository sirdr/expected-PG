from __future__ import division # python 2 compat
import scipy.integrate as integrate
import numpy as np
import time



def compute_integral(f, low, high, shape, prec):
    integral = np.zeros(shape)
    a_range = np.linspace(low, high, (high-low)/prec)
    for a in a_range:
        integral += f(a)*prec
    return integral


###Adaptive Quadrature###
def compute_integral_asr(f, low, high, tol):
    """Uses Adaptive Simpson's Method"""
    return quad_asr(f, low, high, tol)


def _quad_simpsons_mem(f, a, fa, b, fb):
    """Evaluates the Simpson's Rule, also returning m and f(m) to reuse"""
    m = (a+b) / 2
    fm = f(m)
    return (m, fm, abs(b-a) / 6 * (fa + 4 * fm + fb))

def _quad_asr(f, a, fa, b, fb, eps, whole, m, fm):
    """
    Efficient recursive implementation of adaptive Simpson's rule.
    Function values at the start, middle, end of the intervals are retained.
    """
    lm, flm, left  = _quad_simpsons_mem(f, a, fa, m, fm)
    rm, frm, right = _quad_simpsons_mem(f, m, fm, b, fb)
    delta = left + right - whole
    if np.linalg.norm(delta, ord=np.inf) <= 15*eps: #abs(delta) <= 15 * eps:
        return left + right + delta / 15
    return _quad_asr(f, a, fa, m, fm, eps/2, left , lm, flm) +\
           _quad_asr(f, m, fm, b, fb, eps/2, right, rm, frm)


def quad_asr(f, a, b, eps):
    """Integrate f from a to b using Adaptive Simpson's Rule with max error of eps."""
    fa, fb = f(a), f(b)
    m, fm, whole = _quad_simpsons_mem(f, a, fa, b, fb)
    return _quad_asr(f, a, fa, b, fb, eps, whole, m, fm)

##Tests###

#def f(a):
#    mat = np.zeros((3,3))
#    for i in range(0,3):
#        mat[0][i] = np.exp(3*a)*np.sin(a+1)
#    for i in range(0,3):
#        mat[1][i] = np.cos(a+2)
#    for i in range(0,3):
#        mat[2][i]= np.sin(a+3)
#    return mat

#start = time.time()
#result = compute_integral(f, -1, 3, (3,3), .0001)
#end = time.time()
#print(result)
#print(end-start)

#start = time.time()
#result= compute_integral_asr(f, -1, 3, 1e-03)
#end = time.time()
#print(result)
#print(end-start)



####Old###

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
