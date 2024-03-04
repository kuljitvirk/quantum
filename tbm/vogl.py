"""
*** NOT TESTED YET ***

Implementation of the Tight Binding model from the seminal work by Vogl et. al. 
[1] "A Semi-empirical tight-binding theory of the electronic structure of semiconductors",
Journal of Physics and Chemistry of Solids, 365, Vol 44 (1983)

In this paper, Vogl et al. introduced excited state, denoted as s*, on each atom. The
resulting tight binding model produced a fairly accurate density of states for many semiconductors
including Si, Ge, GaAs, and several others.
"""
import numpy as np

# For clarity, I define the states using the same labels as in the paper
# and enumerate them as shown in Table (A) on Page 367 of [1]

# [1] uses 'a' to denote orbitals of the anion and 'c' for orbitals of the cation.
# The functions g0(k), g1(k), g2(k), g3(k) are defined for matrix elements where
# the cation is in the column index and anion in the row index. For example:
# <pax| H | sc > ~ g1
# <sc | H | pax> ~ conj(g1)

# Enumeration of the independent digonal matrix elements, E(.) in [1]
sa, sc, pa, pc, ssa, ssc = 0,1,2,3,4,5

# Enumeration of the independent (non-zero) matrix elements V
s_s, x_x, x_y, sa_pc, sc_pa, ssa_pc, pa_ssc, ss_ss = 0,1,2,3,4,5,6,7

# Enumeration of the 10 states in the full Hamiltonian
sa  = 0
sc  = 1 
pax = 2
pay = 3
paz = 4 
pcx = 5
pcy = 6
pcz = 7
ssa = 8
ssc = 9

nstates = 10

def hamiltonian_matrix(kpoints : np.ndarray, lattice_constant : float, E : np.ndarray, V : np.ndarray):
    """
    Returns the Hamiltonian matrices over a list of wavevectors
    Args:
        kpoints (np.ndarray)     : List of N wavevectors with each row corresponding to a 3-dimensional wavevector
        lattice_constant (float) : The lattice constant (aL parameter) used in the model [1]
        E (np.ndarray)           : 10 energy parameters along the diagonal in the model, enumerated as shown above.
        V (np.ndarray)           : Matrix of independent V parameters in the model with states enumerated as shown above.
    """
    H = np.zeros((len(kpoints), nstates, nstates), dtype=complex)

    aL4 = lattice_constant/4
    coska = np.cos(kpoints*aL4)
    sinka = np.sin(kpoints*aL4)

    g0 = ( coska[:,0] * coska[:,1] * coska[:,2] -
          1j * sinka[:,0] * sinka[:,1] * sinka[:,2])

    g1 = (-coska[:,0] * sinka[:,1] * sinka[:,2] -
          1j * sinka[:,0] * coska[:,1] * coska[:,2])

    g2 = (-sinka[:,0] * coska[:,1] * coska[:,2] -
          1j * coska[:,0] * sinka[:,1] * sinka[:,2])

    g3 = (-sinka[:,0] * sinka[:,1] * coska[:,2] -
          1j * coska[:,0] * coska[:,1] * sinka[:,2])

    # Upper triangular portion
    H[:,  sa,  sc]  = V[s_s]   * g0
    H[:,  sa, pcx]  = V[sa_pc] * g1
    H[:,  sa, pcy]  = V[sa_pc] * g2
    H[:,  sa, pcz]  = V[sa_pc] * g3

    Vpa_sc = V[sc_pa]
    H[:, pax,  sc] = -Vpa_sc * g1
    H[:, pay,  sc] = -Vpa_sc * g2
    H[:, paz,  sc] = -Vpa_sc * g3

    H[:, pax, pcx] = V[x_x] * g0
    H[:, pax, pcy] = V[x_y] * g1
    H[:, pax, pcz] = V[x_y] * g2

    H[:, pay, pcx] = V[x_y] * g3
    H[:, pay, pcy] = V[x_x] * g0
    H[:, pay, pcz] = V[x_y] * g1

    H[:, paz, pcx] = V[x_y] * g2
    H[:, paz, pcy] = V[x_y] * g1
    H[:, paz, pcz] = V[x_x] * g0

    H[:, ssa, pcx] = V[ssa_pc] * g1
    H[:, ssa, pcy] = V[ssa_pc] * g1
    H[:, ssa, pcz] = V[ssa_pc] * g1

    H[:, ssa, ssc] = V[ss_ss] * g0

    H = H + np.conj(H.transpose((0,2,1)))

    # Set the diagonal elements
    H[:,sa,sa]   = E[sa]
    H[:,sc,sc]   = E[sc]
    
    H[:,pax,pax] = E[pa]
    H[:,pay,pay] = E[pa]
    H[:,paz,paz] = E[pa]
    
    H[:,pcx,pcx] = E[pc]
    H[:,pcy,pcy] = E[pc]
    H[:,pcz,pcz] = E[pc]
    
    H[:,ssa,ssa] = E[ssa]
    H[:,ssc,ssc] = E[ssc]

    return H

def band_circuit():
    pass

def bandstructure():
    pass

def dos():
    pass

def optical_absorption():
    pass



    

    