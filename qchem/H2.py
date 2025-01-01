import numpy as np
from scipy import special
from math import pi,sqrt

def correlation(vec : np.ndarray):
    vec = np.asarray(vec)
    alpha = vec[:len(vec)//2]
    coeff = vec[len(vec)//2:]
    S = -2*np.sqrt(alpha) + np.exp(0.25/alpha)*sqrt(pi)*(1 + 2*alpha)*special.erfc(0.5/np.sqrt(alpha))
    S /= 4*2**(1/4)*pi**(5/4)*alpha**(7/4)
    return 4*pi*np.sum( coeff * S)

def gen_sto_ng_basis(self, n : int):
    """
    Fits Slater Type Orbital: (zeta^3/pi)^(1/2) exp(-zeta r) 
    to sum of n-Gaussians, where one Gaussian is of the form
    (2 a /pi)^(3/4)exp(-a r^2)
    """







if __name__=='__main__':
    pass