import numpy as np
from scipy.sparse import coo_matrix

SMALLTOL1 = 1e-8
cmplx_eye = complex(0.,1.)

PAULIX = np.array([[0,1],[1,0]])
PAULIY = np.array([[0,complex(0.,-1)],[complex(1,0),0]])
PAULIZ = np.array([[1.,0.],[0.,-1.]])
PAULII = np.array([[1.,0.],[0.,1.]])

todecimal = lambda statevector : sum([2**(len(statevector)-i-1) * v for i,v in enumerate(statevector)])

def two_level_decomposition(A):
    """
    Decomposes a unitary matrix A into a product of 2-level unitaries
    Args:
        A (ndarray, complex) : Unitary matrix
    Returns:
        Ulist = [U1, U2, U3, ..., Un] such that, A = U1 U2 U3 ... Un
    """
    nrows, ncols = A.shape
    assert nrows==ncols
    # Is A unitary
    Id = np.eye(nrows)
    is_unitary = np.abs( A @ np.conj(A).T - Id ).max() < SMALLTOL1
    assert is_unitary
    Ulist = []
    _A = A.copy()
    for j in range(ncols):
        for i in range(j+1,nrows):
            a = _A[j,j]
            b = _A[i,j]
            h = np.sqrt(a*np.conj(a) + b*np.conj(b))
            U = Id.copy()
            if h > SMALLTOL1:
                U[j,j] = np.conj(a)/h
                U[j,i] = np.conj(b)/h
                U[i,j] = -b/h
                U[i,i] =  a/h
            Ulist += [U]
            _A = U @ _A
    # The loop does not deal with the final diagonal entry
    # That entry can be +1 or -1 and so its inverse is the entry itself
    U = Id.copy()
    U[-1,-1] = _A[-1,-1]
    Ulist += [U]
    # At this point:  Ulist[n-1] ... Ulist[1] Ulist[0] A = I
    # A = inv(Ulist[0]) inv(Ulist[1])... inv(Ulist[n-2]) inv(Ulist[n-1])
    Ulist = [np.conj(u).T for u in Ulist]
    return Ulist

class BitBasis(object):
    def __init__(self, nbits):
        self.nbits = nbits
        self.dim = 2**nbits
        self.basisenum = np.arange(self.dim)
        self.bitenum = nbits - 1 - np.arange(nbits)
        self.bitvalue = 2**self.bitenum
        self.basisvectors = np.array([[int(i) for i in np.binary_repr(v,self.nbits)] for v in self.basisenum],dtype=int)

    def to_basisenum(self, vector):
        return sum(vector * self.bitvalue)

    def to_basisvector(self, n):
        return np.array([int(i) for i in np.binary_repr(n,self.nbits)])


    def bit_pauli_x(self, bit, basisvector):
        """
        Returns: phase, bit flipped state where phase = 1 for pauli x
        """

        bit = self.nbits - 1 - bit
        basisvector_ = basisvector.copy()
        basisvector_[bit] = 1-basisvector[bit]
        return 1, basisvector_

    def bit_pauli_y(self, bit, basisvector):
        bit = self.nbits - 1 - bit
        basisvector_ = basisvector.copy()
        basisvector_[bit] = 1-basisvector[bit]
        return cmplx_eye if basisvector[bit]==0 else -cmplx_eye, basisvector_

    def bit_pauli_z(self, bit, basisvector):
        bit = self.nbits - 1 - bit
        return 1 if basisvector[bit]==0 else -1, basisvector

    def repr_pauli_x1(self, bit):
        cols = np.arange(self.dim)
        rows = np.zeros(self.dim)
        X = np.zeros(self.dim, dtype=float)
        for col in cols:
            a, v = self.bit_pauli_x(bit, self.basisvectors[col])
            row = self.to_basisenum(v)
            # <row| X |col>
            X[col] = a
            rows[col] = row
        X = coo_matrix((X, (rows, cols)),shape=(self.dim, self.dim))
        return X

    def repr_pauli_y1(self, bit):
        cols = np.arange(self.dim)
        rows = np.zeros(self.dim)
        Y = np.zeros(self.dim, dtype=complex)
        for col in cols:
            a, v = self.bit_pauli_y(bit, self.basisvectors[col])
            row = self.to_basisenum(v)
            # <row| X |col>
            Y[col] = a
            rows[col] = row
        Y = coo_matrix((Y, (rows, cols)),shape=(self.dim, self.dim))
        return Y

    def repr_pauli_z1(self, bit):
        cols = np.arange(self.dim)
        rows = np.zeros(self.dim)
        Z = np.zeros(self.dim, dtype=float)
        for col in cols:
            a, v = self.bit_pauli_z(bit, self.basisvectors[col])
            row = self.to_basisenum(v)
            # <row| X |col>
            Z[col] = a
            rows[col] = row
        Z = coo_matrix((Z, (rows, cols)),shape=(self.dim, self.dim))
        return Z

def pauli_matrices(dim):
    pass