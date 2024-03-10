"""
This module implements the tight binding band structure model for Zinc Blende structure.
The diamond structure has the same arrangement except that the atoms in the unit cell are identical species. 
Therefore, the module works for both cases. 

The 8-band model is an implementation of Chapter 2 of [1].
The 10-band model is an implementation of [2]

[1] Yu, Peter and Manuel Cardona, "Fundamentals of Semiconductors"
[2] "A Semi-empirical tight-binding theory of the electronic structure of semiconductors",
Journal of Physics and Chemistry of Solids, 365, Vol 44 (1983)
"""
import numpy as np
import os
from math import pi
import pandas as pd
from collections import namedtuple
from scipy import linalg

lattice_vectors = np.array(
    [[0,1/2,1/2],
     [1/2,0,1/2],
     [1/2,1/2,0]])

reciprocal_lattice_vectors = 2*pi * np.array(
    [[-1, 1, 1],
     [ 1,-1, 1],
     [ 1, 1,-1]])

symmetry_directions_fcc = dict(
    Delta  = (1,0,0),
    Lambda = (1,1,1),
    Sigma  = (1,1,0)
)

bz_points = dict(
    G = 2*pi*np.array((0,0,0)),
    X = 2*pi*np.array((0,1,0)),    
    U = 2*pi*np.array((1/4,1,1/4)),
    W = 2*pi*np.array((1/2,1,0)),  
    K = 2*pi*np.array((3/4,3/4,0)),
    L = 2*pi*np.array((1/2,1/2,1/2)))

# States for the 8-band model
S1, S2, X1, Y1, Z1, X2, Y2, Z2 = 0,1,2,3,4,5,6,7
# Data structure for parameters of the 8-band model
TB08params_t = namedtuple('tbparams', ('Es','Ep','Vss','Vsp','Vxx','Vxy'))

# States for the 10-band model by Vogl et al.
SA, SC, XA, YA, ZA, XC, YC, ZC, SSA, SSC = 0,1,2,3,4,5,6,7,8,9
# Data structure for the parameters for the 10-band model by Vogl et al.
TB10params_t = namedtuple('tbparams10', 
                          ('Esa','Epa','Esc','Epc','Essa','Essc','Vss','Vxx','Vxy','Vsapc','Vscpa','Vssapc','Vpassc'))

TB08params = dict(
    C  = TB08params_t(Es=0., Ep=7.40,  Vss=-15.20, Vsp=10.25, Vxx=3.00, Vxy=8.30),
    Si = TB08params_t(Es=0., Ep=7.20,  Vss= -8.13, Vsp= 5.88, Vxx=3.17, Vxy=7.51),
    Ge = TB08params_t(Es=0., Ep=8.41,  Vss= -6.78, Vsp= 5.31, Vxx=2.62, Vxy=6.82))

TB10params = dict(
    Si = TB10params_t(
        Esa=-4.200,Epa=1.715,Esc=-4.200,Epc=1.715,Essa=6.685,Essc=6.685,
        Vss=-8.300,Vxx=1.715,Vxy=4.575,Vsapc=5.7292,Vscpa=5.7292,
        Vssapc=5.3749,Vpassc=5.3749))

def line(u1,u2, spacing):
    """
    Creates an array of points of the form:
    u1 + (u2-u1) * t, where t = 0...1
    Args:
        u1 : tuple/array representing the point u1 at the start of the line.
        u2 : tuple/arra representing the point u2 at the end of the line
        spacing : the spacing between subsequent points on the line
    """
    du = np.array(u2) - np.array(u1)
    nk = int(np.ceil(np.sqrt((du**2).sum()) / spacing))
    t = np.arange(nk)/nk
    return u1[None, :] + du[None,:] * t[:,None]

def circuit1(dk):
    """
    L --> G --> X --> U, K, --> G
    Returns:
        Array of kpoints with spacing set by dk
        Indices of the special points L, G, X, U, G
    """
    kpt = [
        line(bz_points['L'], bz_points['G'],dk),
        line(bz_points['G'], bz_points['X'],dk),
        line(bz_points['X'], bz_points['U'],dk),
        line(bz_points['K'],bz_points['G'],dk)
    ]
    idx = np.cumsum([0] + [len(k) for k in kpt])
    kpt = np.vstack(kpt)
    return kpt,idx

def phase_factors(kpoint):
    """
    The four phase factors that arise from summing the Hamiltonian over nearest neighbors
    of an atom in the Zincblende structure. 
    """
    # The locations of the nearest neighbors
    dalpha = np.array([
        [ 1, 1, 1],
        [ 1,-1,-1],
        [-1, 1,-1],
        [-1,-1, 1]])/4
    expd = 0.25 * np.exp(1j * dalpha @ kpoint)
    g1 = expd[0] + expd[1] + expd[2] + expd[3]
    g2 = expd[0] + expd[1] - expd[2] - expd[3]
    g3 = expd[0] - expd[1] + expd[2] - expd[3]
    g4 = expd[0] - expd[1] - expd[2] + expd[3]
    return g1, g2, g3, g4

def hamiltonian_tb08(t, kpoint):
    """
    Implementation of the model in Table 2.2 of [1].
    """

    kpoint = np.asarray(kpoint)
    assert kpoint.shape[0]==3

    g1, g2, g3, g4 = phase_factors(kpoint)

    H = np.zeros((8,8), dtype=complex)
    # We define the elements where gi are unconjugated. 
    # The conjugate transpose and then the diagonal part is added below
    H[S1, S2] =  t.Vss * g1
    H[S1, X2] =  t.Vsp * g2
    H[S1, Y2] =  t.Vsp * g3
    H[S1, Z2] =  t.Vsp * g4

    H[X1, S2] = -t.Vsp * g2
    H[Y1, S2] = -t.Vsp * g3
    H[Z1, S2] = -t.Vsp * g4

    H[X1, X2] =  t.Vxx * g1
    H[Y1, X2] =  t.Vxy * g4
    H[Z1, X2] =  t.Vxy * g3

    H[X1, Y2] =  t.Vxy * g4
    H[Y1, Y2] =  t.Vxx * g1
    H[Z1, Y2] =  t.Vxy * g2

    H[X1, Z2] =  t.Vxy * g3
    H[Y1, Z2] =  t.Vxy * g2
    H[Z1, Z2] =  t.Vxx * g1

    H = H + np.conj(H.T)

    E = np.zeros(8, dtype=float)
    E[:2] = t.Es
    E[2:] = t.Ep
    H[np.diag_indices(8)] = E
    return H

def get_vogl_parameters(return_table=False):
    with open(os.path.join(os.path.dirname(__file__),'vogl_tb_parameters.txt')) as file:
        lines = file.readlines()
        edata = dict()
        vdata = dict()
        for line in lines:
            if line.startswith('#'):
                continue
            if not line.strip().startswith('Compound'):
                line = line.split()
                if len(line)==7:
                    edata[line[0]] = [float(d) for d in line[1:]]
                if len(line)==8:
                    vdata[line[0]] = [float(d) for d in line[1:]]
        #
    T = pd.concat([pd.DataFrame(edata),pd.DataFrame(vdata)]).T
    T.columns = TB10params_t._fields
    params = dict()
    for name, row in T.iterrows():
        params[name] = TB10params_t(**row.to_dict())
    if return_table:
        return T
    return params

def hamiltonian_tb10(t, kpoint):

    g0,g1,g2,g3 = phase_factors(kpoint)

    H = np.zeros((10,10), dtype=complex)

    H[SA, SC] =  t.Vss   * g0
    H[SA, XC] =  t.Vsapc * g1
    H[SA, YC] =  t.Vsapc * g2
    H[SA, ZC] =  t.Vsapc * g3

    H[XA, SC] = -t.Vscpa * g1
    H[YA, SC] = -t.Vscpa * g2
    H[ZA, SC] = -t.Vscpa * g3

    H[XA, XC] =  t.Vxx * g0
    H[YA, XC] =  t.Vxy * g3
    H[ZA, XC] =  t.Vxy * g2

    H[XA, YC] =  t.Vxy * g3
    H[YA, YC] =  t.Vxx * g0
    H[ZA, YC] =  t.Vxy * g1

    H[XA, ZC] =  t.Vxy * g2
    H[YA, ZC] =  t.Vxy * g1
    H[ZA, ZC] =  t.Vxx * g0

    H[SSA,  XC] =  t.Vssapc * g1
    H[SSA,  YC] =  t.Vssapc * g2
    H[SSA,  ZC] =  t.Vssapc * g3

    H[XA, SSC] = -t.Vpassc * g1
    H[YA, SSC] = -t.Vpassc * g2
    H[ZA, SSC] = -t.Vpassc * g3

    H = H + np.conj(H.T)

    E = np.zeros(10, dtype=float)
    E[SA] = t.Esa
    E[SC] = t.Esc
    E[[XA,YA,ZA]] = t.Epa
    E[[XC,YC,ZC]] = t.Epc
    E[SSA]        = t.Essa
    E[SSC]        = t.Essc

    H[np.diag_indices(10)] = E

    return H

