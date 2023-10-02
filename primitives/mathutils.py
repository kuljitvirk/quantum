from math import floor
import numpy as np
def to_continued_fractions(A : float,tol=1e-6,maxiter=100):
    a = []
    while 1:
        x = int(floor(A))
        a += [x]
        y = A - x
        if y<tol or len(a)>maxiter:
            break
        A = 1/y
        #print(x,y, A)
    return a
def from_continued_fractions(a : list, return_series = False):       
    p = np.zeros(len(a))
    q = np.zeros(len(a))
    p[0] = a[0]
    p[1] = 1 + a[0]*a[1]
    q[0] = 1
    q[1] = a[1]
    for n in range(2,len(a)):
        p[n] = a[n]*p[n-1] + p[n-2]
        q[n] = a[n]*q[n-1] + q[n-2]
    p = p.astype(int)
    q = q.astype(int)
    if return_series:
        return p, q
    return p[-1],q[-1]
def to_binary(A, nbits):
    B = []
    a = A
    for i in range(nbits):
        B += [a % 2]
        a = a // 2
    # Most significant bit at index 0
    return np.array(B)[::-1]
def to_decimal(A):
    nbits = len(A)
    n = np.arange(nbits)[::-1]
    return np.sum(A * 2**n,axis=0)