from __future__ import division # python 2 compat
import numpy as np
import time
import bisect
from math import cos, sin, sqrt



def compute_integral(f, low, high, shape, prec):
    integral = np.zeros(shape)
    a_range = np.linspace(low, high, (high-low)/prec)
    for a in a_range:
        integral += f(a)*prec
    return integral


###Simpson's Adaptive Quadrature###
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

#Guass-Kronrod Adaptive Quadrature##

# nodes and weights for Gauss-Kronrod
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
    """
    This function computes $\int_a^b \mathrm{d}x f(x)$ using Gauss-Kronrod
    quadrature formula. The integral is transformed
    z  = 2 \\frac{x-a}{b-a}-1
    x  = \\frac{b-a}{2} (z+1) + a
    dz = 2 \\frac{dx}{b-a}
    dx = \\frac{b-a}{2} dz
    \int_a^b \mathrm{d}x f(x) = \\frac{b-a}{2} \int_{-1}^1 \mathrm{d}z f((z+1)*(b-a)/2+a)
    returns integral and an error estimate
    """
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
        
    #integral_G7 = np.sum(integrand[:7]*gauss_weights)
    integral_K15 = np.zeros(integral_G7.shape)
    for i in range(0,len(kronrod_weights)):
        integral_K15 += integrand_list[i]*kronrod_weights[i]
        
    #integral_K15 = np.sum(integrand*kronrod_weights)

    error = (200*np.linalg.norm(integral_G7-integral_K15, ord = np.inf))**1.5

    return integral_K15*dx, dx*error


def integrate(f, a, b, args=(), minintervals=1, limit=200, tol=1e-10):
    """
    Do adaptive integration using Gauss-Kronrod.
    """
    #fv = np.vectorize(f)

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
            return False  # better to raise an exception

        err, left, right, I = intervals.pop()

        # split integral
        mid = left+(right-left)/2

        # calculate integrals and errors, replace one item in the list and
        # append the other item to the end of the list
        I, err = integrate_gausskronrod(f, left, mid, args)
        bisect.insort(intervals, (err, left, mid, I))
        I, err = integrate_gausskronrod(f, mid, right, args)
        bisect.insort(intervals, (err, mid, right, I))


def compute_integral_GQ(f, low, high, tol):
    '''Gauss_Kronrod Adaptive Quadrature'''
    return integrate(f, low, high, tol=tol)[0]

'''
##Tests###
def f(a):
    mat = np.zeros((3,3))
    for i in range(0,3):
        mat[0][i] = np.exp(3*a)*np.sin(a+1)
    for i in range(0,3):
        mat[1][i] = np.cos(a+2)
    for i in range(0,3):
        mat[2][i]= np.sin(a+3)
    return mat
start = time.time()
result1= compute_integral_GQ(f,0,2*np.pi,0.1)
end = time.time()
print(result1)
print(end-start)

start = time.time()
result2 = compute_integral_asr(f,0,2*np.pi,0.1)
end = time.time()
print(result2)
print(end-start)


print(np.allclose(result1, result2))
'''
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
