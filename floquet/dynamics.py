"""
How to create Bloch Floquet Hamiltonian

H = bloch_floquet_hamiltonian( V, num_photons )

V[0] : l =  0 sector or the diagonal block
V[1] : l = -1 sector or the first block diagonal in upper triangular portion
V[2] : l = -2 sector and so on

Corresponding time domain Hamiltonian is:

H(t) = V[0] + V[1] e^{i w t} + hermitian(V[1]) e^{-i w t} + 

"""

import numpy as np
from scipy import linalg
import scipy.special as sp
from math import pi
from numpy import sin, cos, tan
from math import factorial
from ..tbm import tbtmdc

def matrix_exp(a, H, hermitian=True):
    """
    Exponentiation of H by direct diagonalization
    Args:
        a (complex): scalar to multiply H by
        H (complex) : square matrix
    Keywords
        hermitian (boolean) : if true uses linalg.eigh function for diagonalization
    Returns:
        U = exp( a H) 
    """
    if hermitian:
        W, V = linalg.eigh(H)
    else:
        W, V = linalg.eig(H)
    U = V @ np.diag( np.exp( a * W) ) @ V.conj().T
    return U

def fast_matrix_exp(a,H,hermitian=True):
    """
    exp(a H) ~  (1 - a /2 H)^-1 (1 + a/2 H)
    """
    I = np.eye(H.shape[0])
    U = linalg.solve(I - a/2 * H, I + a/2 * H)
    return U

def bloch_pierels_hamiltonian(Hcomp,hopping_vectors,kvec,Avec,phivec,max_order):
    """
    Composes a Hamiltonian matrix as 
    H = (-i)^l Jl(A.d[j]) V(+,j) exp( i k.d[j]) + (-i)^l Jl(-A.d[j]) V(-,j) exp(-i kvec.d[j])
    
    where Jl is the Bessel function order l.
    """
    kvec = np.array(kvec)
    A = np.array(Avec)
    phi = np.array(phivec)
    H1 = np.zeros(Hcomp[0,0].shape,dtype=complex)
    if max_order==0:
        for i in range(H1.shape[0]):
            H1[i,i] = Hcomp[0,0,i,i]
    for s in [0,1]:
        for i in range(len(hopping_vectors)):
            d = (-1)**s * hopping_vectors[i]
            Z = A * d # Ax dx , Ay dy
            if np.abs(Z).max() > 1.e-16:

                K = np.sqrt( Z[0]**2 + Z[1]**2 + 2*Z[0]*Z[1]*cos(phi[0]-phi[1]) )
                cosp = Z @ np.cos(phi) / K
                sinp = Z @ np.sin(phi) / K
                psi = np.arctan2( sinp, cosp )

                f = sp.jv(max_order, A @ d) * np.exp( -1j * max_order * (pi/2 + pi + psi) +  1j* kvec @ d )
            elif max_order == 0:
                f = np.exp( 1j* kvec @ d )     
            else:
                f = 0            
            H1 = H1 + f * Hcomp[s,i+1]
    return H1

def bloch_pierels_blocks(Hcomp, hopping_vectors, kvec, Avec, phi,max_order=1, max_photons=None):
    """
    Returns:
        HX (complex 3D array):  HX[i] = block for making ith diagonal
    """
    if max_photons is None:
        max_photons = max_order
    HD = bloch_pierels_hamiltonian(Hcomp, hopping_vectors,kvec ,Avec, phi,0)
    HX = np.zeros((max_order + 1,) + HD.shape, dtype=complex)
    HX[0] = HD
    for l in range(1,max_order+1):
        HX[l] = bloch_pierels_hamiltonian(Hcomp, hopping_vectors,kvec,Avec, phi,-l)
    return HX

def create_extended_space_matrix(H0,E,order,max_photons):
    Isys = np.eye(H0.shape[0])
    H = linalg.block_diag(*[H0 for n in range(-max_photons,max_photons+1)])
    if np.abs(E)>1.e-16:
        Q = linalg.block_diag(*[-n*Isys for n in range(-max_photons,max_photons+1)])
        H = H + Q
    if order!=0:
        H = np.triu(np.roll(H, order*H0.shape[0],axis=1))
    return H

def bloch_floquet_hamiltonian(HX,max_photons=None,Edrive=0.):
    """
    Constructs the Bloch-Floquet Hamiltonian by default without n*Edrive included
    """
    if max_photons is None:
        max_photons = len(HX)
    HD = create_extended_space_matrix(HX[0], Edrive, 0, max_photons)
    H = np.zeros_like(HD).astype(complex)
    for l in range(1,len(HX)):
        H += create_extended_space_matrix(HX[l], 0, l, max_photons)
    H = H + H.conj().T + HD
    return H

def create_projectors(Nsys,Nph):
    Isys = np.eye(Nsys)
    Osys = np.zeros((Nsys,Nsys))
    Proj = []    
    for n in range(2*Nph+1):
        if n == 0:
            P = [Isys] + [Osys]*(2*Nph)
        elif n < 2*Nph:
            P = [Osys]*n + [Isys] + [Osys]*(2*Nph - n)
        else:
            P = [Osys]*(2*Nph) + [Isys]
        Proj += [linalg.block_diag(*P)]
    return Proj

def propagator(tmax, Hfunc, dt = 0.1, hermitian=True, psi=None):
    """
    Trotter-Suzuki method to compute propagator with the time-ordered exponential of a matrix-valued function
    H(t) = Hfunc(t)
    U(t) = exp(idt H(3dt))exp(idt H(2dt))exp(idt H(dt))exp(idt H(0))
    Args:
        tmax (float) : specifies the time span as [0,tmax]
        Hfunc (callable) : Hfunc(t) must return a matrix at time t, and for each t the matrix must be the same size
    Keywords:
        dt (float) : time-step in the Trotter-Suzuki scheme (see above)
        hermitian (boolean) : passed to matrix_exp
    """
    dt = tmax/np.ceil(tmax/dt)
    time = np.arange(0, tmax+dt/2, dt)
    U = np.eye(Hfunc(time[0]).shape[0]).astype(complex)
    Uprev = U.copy()
    Uprev2 = U.copy()
    resid = []
    if psi is not None:
        psit = np.zeros((len(time),2),dtype=complex)
        psit[0]=psi
    else:
        psit = None
    for i in range(1, len(time)):
        H = Hfunc(time[i-1])
        dt = time[i]-time[i-1]
        Udt = matrix_exp(-1j*dt, H, hermitian=hermitian)
        if psi is not None:
            psit[i] = Udt @ psit[i-1]
        U = Udt @ U
        dUdt = -1j*H @ Uprev
        dUdt_num = (U - Uprev)/dt if i==1 else (U - Uprev2)/2/dt
        resid += [np.abs(dUdt_num-dUdt).max()]
        Uprev2 = Uprev
        Uprev = U
    resid = np.array(resid)

    return U,(psit,time)

def time_domain_Hamiltonian(t,Edrive, HX, rotation=-1):
    """
    Returns time domain Hamiltonian as
    
    H = HX[0] + sum(e^(-rotation * i (n+1) Edrive t) HX[i], i=0...N-1)

    rotation means that HX components are such that they contribute with e^{(rotation) -i w t} factors
    
    where N = len(HX). HX[i] is the (i+1)th diagonal block    
    """
    H = HX[0].copy().astype(complex)
    for n in range(1, len(HX)):
        h = HX[n] * np.exp(-rotation * 1j* n *Edrive*t)
        H += h + h.conj().T
    return H

def quasi_energy_spectrum_time_domain(
    Edrive,
    Hlist,
    Rlist=None,
    amplitude=None,
    phase=None,
    hopping_vectors=None,
    kvec=None,
    rotation=-1, 
    trotter_steps=1000, 
    return_states=False, 
    return_propagator=False):
    """
    Args:
        Edrive: fundamental frequency (energy units)
        Hlist : list of diagonal and off-diagonal blocks, or components for plane wave expansion (see tbtmdc)
    Keywords:
        Rlist : components of "r" operator such that r.E is the same shape as Hlist when using plane wave expansion
                Rlist[0] = r[0] operator for first Cartesian direction etc.
        amplitude : Array of [Ax,Ay] amplitudes
        phase     : Array of phase shifts [phix,phiy]
        trotter_steps : number of discretization steps for Trotter-Suzuki formula
    Returns:
        W (1-d array) : W[i] = 1j*log(u[i]) * E/2pi, 
        
    where u are the eigenvalues of the unitary operator    
    
    Ufull = U(N-1) U(N-2) ... U(1) Id    
    
    where Ufull is the Trotter-Suzuki construction with 
    
    U(n) = exp(-i dt H(n*dt)), 
    
    where H(n*dt) = time_domain_Hamiltonian(t,E,HD,HX),
    where HD,HX are computed from the bloch_floquet_hamiltonian
    
    """
    if amplitude is not None:
        assert not np.isscalar(amplitude) and not np.isscalar(phase)
        assert np.isscalar(Edrive) or len(Edrive)==len(amplitude)
        if np.isscalar(Edrive):
            Edrives = np.array([Edrive])
        Amplitudes = np.atleast_2d(amplitude)
        Phases = np.atleast_2d(phase)

    def Hfunc_floquet_expansion(t):
        return time_domain_Hamiltonian(t, Edrive, Hlist,rotation=rotation)

    def Hfunc_direct(t):
        A = np.zeros(2,dtype=float)
        rE = np.zeros_like(Hlist)
        for i in range(len(Amplitudes)):
            a = Amplitudes[i]*cos(Edrives[i]*t + Phases[i])
            A += a
            rE += (a[0]*Rlist[0] + a[1]*Rlist[1]) * (1j*Edrive)
        return tbtmdc.Hamiltonian_from_comp(Hlist + rE,hopping_vectors,kvec + A)

    Hfunc = Hfunc_floquet_expansion if amplitude is None else Hfunc_direct

    T = 2*pi/Edrive
    U, _ = propagator(T, Hfunc, dt=T/trotter_steps)
    if not return_states:
        W = linalg.eigvals(U)
    else:
        W,V = linalg.eig(U)
    W = 1j*np.log(W)/T
    if np.abs(W.imag).max() < 1.e-6 * np.abs(W).max():
        W = np.real(W)
    if not return_states and not return_propagator:
        return np.sort(W)
    elif (not return_states and return_propagator):
        return np.sort(W),U
    i = np.argsort(W)
    W = W[i]
    V = V[:,i]  
    if not return_propagator:
        return W,V
    return W,V,U

def quasi_energy_spectrum_floquet(Edrive,H,system_dimension,max_photons,return_states=False):
    """
    Args:
        H : full Bloch-Floquet Hamiltonian matrix
        system_dimension : number of states in the original system
        max_photons : maximum number of photon occupation (2*max_photons + 1)*system_dimension = H.shape[0]
    Returns:
        W : eigenvalues
    """
    assert (2*max_photons + 1)*system_dimension == H.shape[0]
    Isys = np.eye(system_dimension)
    Q0   = linalg.block_diag(*[-n*Isys for n in range(-max_photons, max_photons+1)])
    if not return_states:
        W    = np.linalg.eigvalsh(H + Edrive*Q0)
    else:
        W,V = np.linalg.eigh(H + Edrive*Q0)
    if not return_states:
        return np.sort(W)
    i = np.argsort(W)
    W = W[i]
    V = V[:,i]
    return W, V

#==============================================================================    

def decomp_proj(A,Proj):
    """
    Returns:
        AD = sum( p A p, p in Proj)
        A - AD : the complimentary
    """
    AD = np.zeros(A.shape,dtype=complex)
    for p in Proj:
        AD += p @ A @ p
    return AD, A - AD

def bloch_floquet_block_diagonalization_4(Edrive, HD, HX, degen=False):
    """
    4th order block diagonalization of a matrix in the extended Hilbert space
    performed via perturbation theory.
    Diagonal blocks are all of the form
    HD + n Edrive
    HX = List of diagonal blocks at 1, 2, 3, ...
    """
   
    Nph  = len(HX)
    Nsys = HD.shape[0]
    Isys = np.eye(Nsys)
    Osys = np.zeros((Nsys,Nsys))

    VX = np.zeros( ((2*Nph+1) * Nsys,)*2, dtype=complex)
    Q0 = linalg.block_diag(*[-n*Edrive*Isys for n in range(-Nph, Nph+1)])
    VD = linalg.block_diag(*[HD for _ in range(-Nph, Nph+1)])
    E0  = np.diag(Q0)

    if not degen:
        VD = VD + Q0
        E0 = np.diag(VD)
        Q0 = np.diag(E0)
        VD = VD - Q0

    for i, hx in enumerate(HX):
        h = linalg.block_diag(*([hx]*(2*Nph+1)))
        VX += np.triu(np.roll(h, (i+1)*Nsys, axis=1))
    VX = VX + VX.conj().T
    invEMAT = E0 - E0[:,None]

    temp = invEMAT
    invEMAT = invEMAT/(invEMAT + 1.e-16)**2

    C = lambda A,B : A @ B - B @ A
    Proj = []    
    for n in range(2*Nph+1):
        if n == 0:
            P = [Isys] + [Osys]*(2*Nph)
        elif n < 2*Nph:
            P = [Osys]*n + [Isys] + [Osys]*(2*Nph - n)
        else:
            P = [Osys]*(2*Nph) + [Isys]
        Proj += [linalg.block_diag(*P)]
       
    # exp(iG) H exp(-iG) = H + i[G,H] + (i)^2/2! [G,[G,H]] + (i^3)/3! [G,[G,[G,H]]] + i^4/4! [G,[G,[G,[G,H]]]]
    # Order:  Expression
    # 1    : V + i[G1,V ]
    # 2    : i[G2,H0] + i[G1,V] + (i)^2/2 [G1,[G1,H0]] 
    # 3    : i[G3,H0] + i[G2,V] + (i)^2/2 ( [G1,[G2,H0]] + [G2,[G1,H0]] + [G1,[G1,V]] ) + (i^3)/3! [G1,[G1,[G1,H0]]] 
    # 4    : i[G4,H0] + i[G3,V] + (i)^2/2! ( [G1,[G3,H0]] + [G3,[G1,H0]] + [G2,[G1,V]] + [G1,[G2,V]] + [G2,[G2,H0]] ) 
    #                           + (i^3)/3! ( [G1,[G1,[G1,V]]] + [G1,[G2,[G1,H0]]] + [G1,[G1,[G2,H0]]] + [G2,[G1,[G1,H0]]] )

    H0 = Q0
    V = VD + VX

    H1 = VD
    T  = VX
    G1 = 1j * invEMAT * decomp_proj(T, Proj)[1]

    i0 = Nph*Nsys

    T = 1j * C(G1, V) + (1j)**2/factorial(2) * C(G1, C(G1, H0))
    G2 = 1j * invEMAT * decomp_proj(T, Proj)[1]


    T = 1j * C(G2, V) + (1j)**2/factorial(2) * ( C(G1, C(G2, H0)) + C(G2, C(G1, H0)) + C(G1, C(G1, V))) + (1j)**3/factorial(3) * C(G1,C(G1,C(G1,H0)))
    G3 = 1j * invEMAT * decomp_proj(T, Proj)[1]

    T = 1j * C(G3, V) + (1j)**2/factorial(2) * ( C(G1, C(G3, H0)) + C(G3, C(G1, H0)) + C(G2, C(G1, V)) + C(G1, C(G2, V)) + C(G2, C(G2, H0)) ) + (1j)**4/factorial(4) * C(G1,C(G1,C(G1,C(G1,H0))))
    G4 = 1j * invEMAT * decomp_proj(T, Proj)[1]

    U = matrix_exp(-1j , G1 + G2 + G3 + G4)

    H = U.conj().T @ ( H0 + VD + VX  ) @ U
    Heff = H[i0:(i0+Nsys), i0:(i0+Nsys)]

    return Heff, H, U, [G1,G2,G3,G4], [H0 , VD ,  VX], Proj