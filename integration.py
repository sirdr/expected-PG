from __future__ import division 
from math import cos, sin, sqrt
import bisect
import numpy as np
import time



###Fixed-grid Integration Method###
#####################################################################################
def compute_integral(f, low, high, shape, prec):
    '''Simple fixed grid integration estimate '''
    integral = np.zeros(shape)
    a_range = np.linspace(low, high, (high-low)/prec)
    for a in a_range:
        integral += f(a)*prec
    return integral
####################################################################################


###Simpson's Adaptive Quadrature###
###################################################################################
def compute_integral_asr(f, low, high, tol):
    """Interface Function, calls Adaptive Simpson's Method"""
    return quad_asr(f, low, high, tol)


def _quad_simpsons_mem(f, a, fa, b, fb):
    """Calculate  Simpson's Rule"""
    m = (a+b) / 2
    fm = f(m)
    return (m, fm, abs(b-a) / 6 * (fa + 4 * fm + fb))

def _quad_asr(f, a, fa, b, fb, eps, whole, m, fm):
    """Recursively apply adaptive Simpson's rule."""
    lm, flm, left  = _quad_simpsons_mem(f, a, fa, m, fm)
    rm, frm, right = _quad_simpsons_mem(f, m, fm, b, fb)
    delta = left + right - whole
    if np.linalg.norm(delta, ord=np.inf) <= 15*eps: 
        return left + right + delta / 15
    return _quad_asr(f, a, fa, m, fm, eps/2, left , lm, flm) +\
           _quad_asr(f, m, fm, b, fb, eps/2, right, rm, frm)


def quad_asr(f, a, b, eps):
    """Estimate integral of f on interval from a to b using Adaptive Simpson's Method
    with eps as max error"""
    fa, fb = f(a), f(b)
    m, fm, whole = _quad_simpsons_mem(f, a, fa, b, fb)
    return _quad_asr(f, a, fa, b, fb, eps, whole, m, fm)


########################################################################################


#Guass-Kronrod Adaptive Quadrature##
########################################################################################

### Predefined nodes and weights for Gauss-Kronrod as referenced in paper###
gausskronrod_nodes = np.array([0.949107912342759, -0.949107912342759,
                               0.741531185599394, -0.741531185599394,
                               0.405845151377397, -0.405845151377397,
                               0.000000000000000,
                               0.991455371120813, -0.991455371120813,
                               0.864864423359769, -0.864864423359769,
                               0.586087235467691, -0.586087235467691,
                               0.207784955007898, -0.207784955007898])
gauss_weights = np.array([0.129484966168870, 0.129484966168870,
                          0.279705391489277, 0.279705391489277,
                          0.381830050505119, 0.381830050505119,
                          0.417959183673469])
kronrod_weights = np.array([0.063092092629979, 0.063092092629979,
                            0.140653259715525, 0.140653259715525,
                            0.190350578064785, 0.190350578064785,
                            0.209482141084728,
                            0.022935322010529, 0.022935322010529,
                            0.104790010322250, 0.104790010322250,
                            0.169004726639267, 0.169004726639267,
                            0.204432940075298, 0.204432940075298])


def integrate_gausskronrod(f, a, b, args=()):
    """Compute integral using Gauss-Kronrod quadrature method."""
    assert b > a
    mid = 0.5*(b+a)
    dx = 0.5*(b-a)
    zi = mid+gausskronrod_nodes*dx
    integrand_list = [0]*len(zi)
    for i in range(0,len(zi)):
        input = np.ndarray((1,), buffer=np.array([zi[i]]))
        integrand_list[i] = f(input)
    integral_G7 = np.zeros(integrand_list[0].shape)
    for i in range(0,7):
        integral_G7 += integrand_list[i]*gauss_weights[i]
        
    integral_K15 = np.zeros(integral_G7.shape)
    for i in range(0,len(kronrod_weights)):
        integral_K15 += integrand_list[i]*kronrod_weights[i]
        
    error = (200*np.linalg.norm(integral_G7-integral_K15, ord = np.inf))**1.5
    return integral_K15*dx, dx*error


def integrate(f, a, b, args=(), minintervals=1, limit=200, tol=1e-10):
    """Adaptively integrate using Gauss-Kronrod."""
    intervals = []

    limits = np.linspace(a, b, minintervals+1)
    for left, right in zip(limits[:-1], limits[1:]):
        I, err = integrate_gausskronrod(f, left, right, args)
        bisect.insort(intervals, (err, left, right, I))

    while True:
        Itotal = sum([x[3] for x in intervals])
        err2 = sum([x[0]**2 for x in intervals])
        err = sqrt(err2)
        
        if abs(err/np.linalg.norm(Itotal,ord=np.inf)) < tol:
            return Itotal, err

        # no convergence
        if len(intervals) >= limit:
            return False  

        err, left, right, I = intervals.pop()

        # split integral
        mid = left+(right-left)/2

        I, err = integrate_gausskronrod(f, left, mid, args)
        bisect.insort(intervals, (err, left, mid, I))
        I, err = integrate_gausskronrod(f, mid, right, args)
        bisect.insort(intervals, (err, mid, right, I))


def compute_integral_GQ(f, low, high, tol):
    '''Gauss_Kronrod Adaptive Quadrature'''
    return integrate(f, low, high, tol=tol)[0]
#######################################################################################
