"""
Implements quantum primitives:

QFT : numpy fft is used to transform lists of states
Period Finding: Uses QFTbasis to implement probabilistic period finding following the standard algorithm
Order Finding : Calls period finding with the specialized function for order 
Shor Algorithm: Factorizes numbers using Shor algorithm and order finding subroutine above
"""
import numpy as np
from scipy import fft
from math import pi
from .mathutils import to_continued_fractions, from_continued_fractions

TOL1 = 1e-12

def BasisVector(i : int, N : int):
    v = np.zeros(N, dtype=int)
    v[i]=1
    return v

def BasisSet(N : int):
    v = np.eye(N,dtype=int)
    return v

def QFT(x):
    """
    N = len(x)
    y[k] = sum( x * exp( 2j pi k arange(N)/N) )
    """
    psif = fft.fft(np.conj(x)).conj()
    if np.abs(psif.imag).max() < TOL1:
        psif = np.real(psif)
    return psif

def exptPeriodFinding(func, N, phase_noise=None):
    """
    Args:
        func : Function whose period is being sought
        N    : Size of group
    Keywords:
        phase_noise : Applied as exp(1j * phase_noise[i] ) to ith basis vector's contribution to QFT
    Returns:
        k, P where
        k    : Value obtained after projective measurement of the first register
        P    : Probability distribution over values from which k is sampled (hidden information)
    """
    basis  = np.array([(x,func(x)) for x in range(N)])
    psi    = np.ones(N)/np.sqrt(N)
    # Measure the second register
    rng = np.random.default_rng()
    indx = np.arange(N)
    i = rng.choice(indx, 1)
    f = basis[i,1]
    basis = basis[basis[:,1]==f]
    # Hidden variable
    A = basis.shape[0]
    psi = np.ones(A)/np.sqrt(A)
    # We do not need the second register anymore
    basis = basis[:,0].astype(int)
    # Phase noise is just = 0 if not specified, i.e. no noise
    exp_phase_noise = np.ones(len(basis)) if phase_noise is None else np.exp(1j*phase_noise)
    y = np.zeros(N,dtype=complex)
    for i, b in enumerate(basis):
        v  = BasisVector(b, N)
        y += QFT(v)*psi[i] * exp_phase_noise[i]
    
    psi = y/np.sqrt(N)
    P = np.abs(psi)**2
    # Measure the register
    k = rng.choice(np.arange(len(P)), 1, p=P)[0]
    return k, P

def find_period(func, N, phase_noise = None, return_hidden = False, maxruns=10):
    """
    Args:
        func : function whose period is sought
        N    : Size of the group (i.e. integers 0....N-1)
    Keywords:
        return_hidden : if hidden information used to simulate experiment is needed
        maxruns       : Maximum number of trials to get probabilistic answer

    Returns:
        period : period of the function
        k    : value of first register after projective measurement
        number of experiments run before correct answer is found (easy to check given func)

        Optional
        (P,p, q) where
        P : Probability distribution from exptPeriodFinding function above
        p,q : Array from continued fraction expansion

    """
    bound = np.sqrt(N)
    ans = int(1)
    check = np.arange(10)
    for nexpt in range(maxruns):
        k,P = exptPeriodFinding(func,N,phase_noise=phase_noise)
        if k  > 0:
            a = to_continued_fractions(k/N)
            p,q = from_continued_fractions(a,return_series=True)
            sel = q < bound
            p = p[sel].astype(int)
            q = q[sel].astype(int)
            s = p /q 
            sel = np.abs(s-k/N) < 1
            if sum(sel)>0:
                l = p[sel][-1]
                a = q[sel][-1]
                per = a
                ans = per
                if np.all(func(check+ans)==func(check)):
                    break
    period = ans
    if return_hidden:
        return period,k,nexpt+1, (P, p, q)
    return period, k, nexpt+1

def power_mod(x, p, M):
    """
    Efficient calculation of (x^p) mod M
    """
    b = np.array([int(i) for i in np.binary_repr(p)])[::-1]
    L = x % M
    X = 1
    for _, zt in enumerate(b):
        if zt>0:
            X *= L
            X = X % M
        L = (L**2) % M
    return X

def find_order_brute_force(x,p,N):
    for i in range(1,N):
        if power_mod(x, i, p) == 1:
            return i
    return np.nan

def find_order(x,p,N):
    def func(j):
        if np.isscalar(j):
            return power_mod(x, j, p)
        return np.array([power_mod(x,i,p) for i in j])
    _,period, _ = find_period(func, N)
    return period

def gcd(x1,x2):
    """
    Euclid's algorithm. Implementation much slower than numpy. Why?
    """
    while x1 != x2 and x1 != 0 and x2 != 0:
        if x1 > x2:
            x1 = x1 % x2
        else:
            x2 = x2 % x1
    return x1 if x1 != 0 else x2

def shor(M,N=1024,maxruns=1000):
    """
    find_order is the only quantum part: it is brute force yet
    """
    if M % 2 == 0:
        return 2,0
    #N = 2**int(2*np.log2(M)+1)
    for nexpt in range(maxruns):
        x = np.random.choice(np.arange(1,M),1)[0]        
        f = np.gcd(x,M)
        if f > 1 and f < M:
            return f, nexpt, 'A. gcd({},M)'.format(x), 0
        r = find_order(x,M,N)
        if r % 2 == 0:
            continue
        # (x^(r/2)-1)(x^(r/2)+1) = x^r - 1 = kM
        y = x**(r//2)
        f = np.gcd(y-1, M)
        if f > 1 and f < M:
            return f,nexpt, 'B. gcd({},M)'.format(y-1), 0
        f = np.gcd(y+1,M)
        if f > 1 and f < M:
            return f,nexpt, 'C. gcd({},M)'.format(y+1), 0
    return (x,r),nexpt, 6



    

    