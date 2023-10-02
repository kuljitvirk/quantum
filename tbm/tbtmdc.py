"""
11-band Tight binding model for Transition Metal Dichalcogenides
"""
import pandas as pd
import numpy as np
from numpy import cos, exp, sin, sqrt
from math import pi
from itertools import product

xhat = np.array([1,0])
yhat = np.array([0,1])

def ehat(i,N):
    e = np.zeros(N)
    e[i] = 1
    return e

basis = ['dxz(o)', 'dyz(o)', 'pz(o)', 'px(o)', 'py(o)', 'dz2(e)', 'dxy(e)', 'dx2-y2(e)', 'pz(e)', 'px(e)', 'py(e)']
num2basis = dict([(i,b) for i, b in enumerate(basis)])
basis2num = dict([(b,i) for i, b in enumerate(basis)])
basis2vec = {}
for key,val in num2basis.items():
    basis2vec[val] = ehat(key, 11)

gammaStates = { 
    'd+2' : (('dx2-y2(e)','dxy(e)'), ( 1/sqrt(2.),  1j/sqrt(2.))),
    'd-2' : (('dx2-y2(e)','dxy(e)'), ( 1/sqrt(2.),-1j/sqrt(2.))),
    'd+1' : (('dxz(o)','dyz(o)'),     (-1/sqrt(2.), 1j/sqrt(2.))),
    'd-1' : (('dxz(o)','dyz(o)'),     ( 1/sqrt(2.),-1j/sqrt(2.))),
    'd0(e)'  : (('dz2(e)',) , (1,)),
    'p+1(e)' : (('px(e)','py(e)'),(-1/sqrt(2.), 1j/sqrt(2.))),
    'p-1(e)' : (('px(e)','py(e)'),( 1/sqrt(2.),-1j/sqrt(2.))),
    'p+1(o)' : (('px(o)','py(o)'),(-1/sqrt(2.), 1j/sqrt(2.))),
    'p-1(o)' : (('px(o)','py(o)'),( 1/sqrt(2.),-1j/sqrt(2.))),
    'p0(o)'  : (('pz(o)',) ,(1,)),
    'p0(e)'  : (('pz(e)',) ,(1,))
}
num2gamma = dict([(i,b) for i,b in enumerate(gammaStates)])
gamma2num = dict([(b,i) for i,b in enumerate(gammaStates)])

def torep(state):
    n = len(gammaStates[state][0])
    return np.array([basis2vec[gammaStates[state][0][i]]*gammaStates[state][1][i] for i in range(n)]).sum(axis=0)

def load_tbtable(name):
    T = pd.read_csv(name, index_col=0, comment='#')
    return T

def tbtable_hterms(table,column):
    """
    """
    if column not in table.columns:
        print('Column {} does not exist in table with columns {}'.format(column,table.columns))
    N = int(max(np.nanmax(table.bra), np.nanmax(table.ket)))
    terms = [t for t in table.term.unique() if t != 'n']
    Hterms = dict([(name, np.zeros((N+1,N+1))) for name in terms])
    for _, row in table.iterrows():
        if row.term != 'n':
            Hterms[row.term][int(row.bra), int(row.ket)] = row[column]
    Hterms['alatt'] = table.loc['alatt',column]
    Hterms['clatt'] = table.loc['clatt',column]   
    return Hterms

def symterms(Hterms):
    zr = np.zeros_like(Hterms['t1'])
    t1 = Hterms['t1'] if 't1' in Hterms else np.zeros_like(Hterms['t1'])
    t2 = Hterms['t2'] if 't2' in Hterms else np.zeros_like(Hterms['t1'])
    t3 = Hterms['t3'] if 't3' in Hterms else np.zeros_like(Hterms['t1'])
    t4 = Hterms['t4'] if 't4' in Hterms else np.zeros_like(Hterms['t1'])
    t5 = Hterms['t5'] if 't5' in Hterms else np.zeros_like(Hterms['t1'])
    t6 = Hterms['t6'] if 't6' in Hterms else np.zeros_like(Hterms['t1'])

    # (α=1,β=2),(α=4,β=5,γ=3), (α = 7,β = 8,γ = 6), (α = 10,β = 11,γ = 9) 
    for a,b,g in [(1,2,None),(4,5,3),(7,8,6),(10,11,9)]:
        t2[a,a] = 1/4 * t1[a,a] + 3/4 * t1[b,b]
        t2[b,b] = 3/4 * t1[a,a] + 1/4 * t1[b,b]
        t2[a,b] =  sqrt(3.)/4 * (t1[a,a] -  t1[b,b]) - t1[a,b]
        t3[a,b] = -sqrt(3.)/4 * (t1[a,a] -  t1[b,b]) - t1[a,b]
        if g is not None:
            t2[g,g] = t1[g,g]
            t2[g,b] =  sqrt(3.)/2 * t1[g,a] - 1/2 * t1[g,b]
            t3[g,b] = -sqrt(3.)/2 * t1[g,a] - 1/2 * t1[g,b]
            t2[g,a] = 1/2 * t1[g,a] + sqrt(3.)/2 * t1[g,b]
            t3[g,a] = 1/2 * t1[g,a] - sqrt(3.)/2 * t1[g,b]
    # (α=1,β=2,α′ =4,β′ =5,γ′ =3),(α= 7,β = 8,α′ = 10,β′ = 11,γ′ = 9)
    for (a,b,ap,bp,gp) in [(1,2,4,5,3),(7,8,10,11,9)]:
        t4[ap,a] =  1/4 * t5[ap,a] + 3/4 * t5[bp,b]
        t4[bp,b] =  3/4 * t5[ap,a] + 1/4 * t5[bp,b]
        t4[bp,a] =  t4[ap,b] = -sqrt(3.)/4 * t5[ap,a] + sqrt(3.)/4 * t5[bp,b]
        t4[gp,a] = -sqrt(3.)/2 * t5[gp,b]
        t4[gp,b] = -1/2 * t5[gp,b]
        t4[9, 6] =  t5[9,6]
        t4[10,6] = -sqrt(3.)/2 * t5[11,6]
        t4[11,6] = -1/2 * t5[11,6]
    Hterms['t2'] = t2
    Hterms['t3'] = t3
    Hterms['t4'] = t4
    Hterms['t5'] = t5
    Hterms['t6'] = t6
    return Hterms


def kdot(kvec, D):
    for i, d in enumerate(D):
        print('k.d{:d} = '.format(i+1), '{:+.2f} π/3'.format( kvec @ d * 3/pi ))
    
def hopping_vectors(Hterms):
    a1 = Hterms['alatt'] * xhat
    a2 = Hterms['alatt'] * ( -1/2 * xhat + sqrt(3.)/2 * yhat)
    d1 = a1
    d2 = a1 + a2
    d3 = a2
    d4 = -( 2 * a1 + a2 )/3
    d5 =  ( a1 + 2 * a2 )/3
    d6 =  ( a1 - a2 )/3
    d7 = -2*( a1 + 2 * a2)/3
    d8 =  2*( 2*a1 + a2 )/3
    d9 =  2*( a2 - a1 )/3
    return [d1,d2,d3,d4,d5,d6,d7,d8,d9]

def lattice_vectors(Hterms):
    """
    Returns:
    a1,a2 : Direct lattice vectors
    b1,b2 : Reciprocal lattice vectors
    M, K : special RLV points
    """
    a1 = Hterms['alatt'] * xhat
    a2 = Hterms['alatt'] * ( -1/2 * xhat + sqrt(3.)/2 * yhat)
    b1 = 2*pi/Hterms['alatt'] * ( xhat + 1/sqrt(3.) * yhat)
    b2 = 4*pi/sqrt(3.)/Hterms['alatt'] * yhat
    G = np.zeros(2)
    M = 1/2 * b1
    K = 1/3. * ( 2*b1 - b2 )
    return (a1,a2), (b1,b2), (M,K)


def Hamiltonian(kvec,Hterms, nn=None):
    """
    Hterms assumed to store matrices in 1-based indexing, thus putting a row/column of zeros at the start.
    """
    kvec = np.array(kvec)
    assert kvec.ndim==1 and kvec.shape[0]==2
    [d1,d2,d3,d4,d5,d6,d7,d8,d9] = hopping_vectors(Hterms)
    
    H = np.zeros(Hterms['E'].shape,dtype=complex)
    Hterms = symterms(Hterms)
    t1 = Hterms['t1'] if 't1' in Hterms else np.zeros_like(Hterms['t1'])
    t2 = Hterms['t2'] if 't2' in Hterms else np.zeros_like(Hterms['t1'])
    t3 = Hterms['t3'] if 't3' in Hterms else np.zeros_like(Hterms['t1'])
    t4 = Hterms['t4'] if 't4' in Hterms else np.zeros_like(Hterms['t1'])
    t5 = Hterms['t5'] if 't5' in Hterms else np.zeros_like(Hterms['t1'])
    t6 = Hterms['t6'] if 't6' in Hterms else np.zeros_like(Hterms['t1'])

    for i in range(H.shape[0]):
        H[i,i] = Hterms['E'][i,i] + 2 * t1[i,i] * cos(kvec @ d1) + 2 * t2[i,i] * ( cos(kvec @ d2) + cos(kvec @ d3) )
    
    for i,j in [(3,5),(6,8),(9,11)]:
        H[i,j] =  2 * t1[i,j] * cos(kvec @ d1)  + t2[i,j] * ( exp(-1j* kvec @ d2) + exp(-1j* kvec @ d3)) + t3[i,j] * ( exp(1j* kvec @ d2) + exp(1j* kvec @ d3))
    
    for i,j in [(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]:
        H[i,j] = -2j * t1[i,j] * sin(kvec @ d1) + t2[i,j] * ( exp(-1j* kvec @ d2) - exp(-1j* kvec @ d3)) + t3[i,j] * (-exp(1j* kvec @ d2) + exp(1j* kvec @ d3))

    for i,j in [(3,1),(5,1), (4,2), (10,6), (9,7), (11,7), (10,8)]:
        H[i,j] = t4[i,j] * ( exp(1j* kvec @ d4) - exp(1j* kvec @ d6) )

    for i,j in [(4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8)]:
        H[i,j] = t4[i,j] * ( exp(1j* kvec @ d4) + exp(1j* kvec @ d6 ) ) + t5[i,j] * exp(1j* kvec @ d5)   

    H2 = np.zeros_like(H)

    H2[ 9, 6] = t6[ 9, 6] *              (  exp(1j* kvec @ d7) + exp(1j* kvec @ d8)   + exp(1j* kvec @ d9) )
    H2[11, 6] = t6[11, 6] *              (  exp(1j* kvec @ d7) - exp(1j* kvec @ d8)/2 - exp(1j* kvec @ d9)/2 )
    H2[10, 6] = t6[11, 6] * sqrt(3.)/2 * ( -exp(1j* kvec @ d8) + exp(1j* kvec @ d9) )
    H2[ 9, 8] = t6[ 9, 8] *              (  exp(1j* kvec @ d7) - exp(1j* kvec @ d8)/2 - exp(1j* kvec @ d9)/2 )
    H2[ 9, 7] = t6[ 9, 8] * sqrt(3.)/2 * ( -exp(1j* kvec @ d8) + exp(1j* kvec @ d9) )
    H2[10, 7] = t6[11, 8] * 3/4        * (  exp(1j* kvec @ d8) + exp(1j* kvec @ d9) )
    H2[11, 7] = t6[11, 8] * sqrt(3.)/4 * (  exp(1j* kvec @ d8) - exp(1j* kvec @ d9) )
    H2[10, 8] = t6[11, 8] * sqrt(3.)/4 * (  exp(1j* kvec @ d8) - exp(1j* kvec @ d9) )
    H2[11, 8] = t6[11, 8] *              (  exp(1j* kvec @ d7) + exp(1j* kvec @ d8)/4 + exp(1j* kvec @ d9)/4 )

    H2 = H2 + H2.conj().T
    H0 = np.diag(np.diag(H))
    H1 = H - H0
    H1 = H1 + H1.conj().T
    H  = H0 + H1 
    if nn is not None:
        return H[1:][:,1:] if nn==1 else H2[1:][:,1:]
    return H[1:][:,1:] + H2[1:][:,1:]

def Hamiltonian_from_comp(Hcomp,dhop,kvec):
    Ht = np.zeros(Hcomp[0,0].shape,dtype=complex)
    for i in range(Ht.shape[0]):
        Ht[i,i] = Hcomp[0,0,i,i]
    for i,d in enumerate(dhop):
        Ht += Hcomp[0,i+1]*np.exp( 1j* kvec @ d ) + Hcomp[1,i+1]*np.exp(-1j* kvec @ d )
    return Ht

def gradk_Hamiltonian(Hcomp,dhop,kvec):
    V = np.zeros((2,)+Hcomp[0,0].shape,dtype=complex)
    for i,d in enumerate(dhop):
        for ic in range(kvec.shape[0]):
            V[ic] += 1j*d[ic]*( Hcomp[0,i+1]*np.exp( 1j* kvec @ d ) - Hcomp[1,i+1]*np.exp(-1j* kvec @ d ) )
    return V

def HamiltonianComp(Hterms, nn=None):
    """
    Hterms assumed to store matrices in 1-based indexing, thus putting a row/column of zeros at the start.
    H = Hcomp[0,a] * exp(1j * kvec @ v[a]) + Hcomp[1,a] * exp(-1j * kvec @ v[a])
    """
    Hterms = symterms(Hterms)
    t1 = Hterms['t1'] if 't1' in Hterms else np.zeros_like(Hterms['t1'])
    t2 = Hterms['t2'] if 't2' in Hterms else np.zeros_like(Hterms['t1'])
    t3 = Hterms['t3'] if 't3' in Hterms else np.zeros_like(Hterms['t1'])
    t4 = Hterms['t4'] if 't4' in Hterms else np.zeros_like(Hterms['t1'])
    t5 = Hterms['t5'] if 't5' in Hterms else np.zeros_like(Hterms['t1'])
    t6 = Hterms['t6'] if 't6' in Hterms else np.zeros_like(Hterms['t1'])

    Hcomp = np.zeros((2,10,)+Hterms['E'].shape,dtype=complex)
    
    for i in range(12):
        Hcomp[0,0,i,i] = Hterms['E'][i,i]
        for s in range(2):
            Hcomp[s,[1,2,3],i,i] = [ t1[i,i] , t2[i,i] , t2[i,i] ]

    for i,j in [(3,5),(6,8),(9,11)]:
        Hcomp[0,[1,2,3],i,j] = [ t1[i,j] ,  t3[i,j], t3[i,j] ]
        Hcomp[1,[1,2,3],i,j] = [ t1[i,j] ,  t2[i,j], t2[i,j] ]
        for s in [0,1]:
            Hcomp[s,[1,2,3],j,i] = np.conj(Hcomp[1-s,[1,2,3],i,j])

    for i,j in [(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]:
        Hcomp[0,[1,2,3],i,j] = [ -t1[i,j], -t3[i,j],  t3[i,j] ]
        Hcomp[1,[1,2,3],i,j] = [  t1[i,j],  t2[i,j], -t2[i,j] ]
        for s in [0,1]:
            Hcomp[s,[1,2,3],j,i] = np.conj(Hcomp[1-s,[1,2,3],i,j])

    for i,j in [(3,1),(5,1), (4,2), (10,6), (9,7), (11,7), (10,8)]:
        Hcomp[0,[4,6],i,j] = [ t4[i,j], -t4[i,j] ]
        for s in [0,1]:
            Hcomp[s,[4,6],j,i] = np.conj(Hcomp[1-s,[4,6],i,j])

    for i,j in [(4,1), (3,2), (5,2), (9,6), (11,6), (10,7), (9,8), (11,8)]:
        Hcomp[0,[4,5,6],i,j] = [ t4[i,j], t5[i,j], t4[i,j] ] 
        for s in [0,1]:
            Hcomp[s,[4,5,6],j,i] = np.conj(Hcomp[1-s,[4,5,6],i,j])

    arr = np.array
    Hcomp[0,[7,8,9], 9, 6] = t6[ 9, 6] * arr([1,1,1])
    Hcomp[0,[7,8,9],11, 6] = t6[11, 6] * arr([1,-1/2,-1/2])
    Hcomp[0,[7,8,9],10, 6] = t6[11, 6] * sqrt(3.)/2 * arr([0, -1, 1])
    Hcomp[0,[7,8,9], 9, 8] = t6[ 9, 8] *              arr([1,-1/2,-1/2])
    Hcomp[0,[7,8,9], 9, 7] = t6[ 9, 8] * sqrt(3.)/2 * arr([0, -1, 1])
    Hcomp[0,[7,8,9],10, 7] = t6[11, 8] * 3/4        * arr([0, 1, 1])
    Hcomp[0,[7,8,9],11, 7] = t6[11, 8] * sqrt(3.)/4 * arr([0, 1, -1])
    Hcomp[0,[7,8,9],10, 8] = t6[11, 8] * sqrt(3.)/4 * arr([0, 1, -1])
    Hcomp[0,[7,8,9],11, 8] = t6[11, 8] *              arr([1, 1/4, 1/4])

    for c in [7,8,9]:
        for i in [9,10,11]:
            for j in [6,7,8]:
                Hcomp[1,c,j,i] = np.conj(Hcomp[0,c,i,j])

    return Hcomp[:,:,1:,1:]