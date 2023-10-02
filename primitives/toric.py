import numpy as np

Zphase = [1,-1]

def lattice(m, periodic=True):
    """
    Creates an m x m lattice with coordination number of 4, i.e. rectangular topology
    Each edge is counted only once, (i,i+1), and (i+1,i) and this is thus a directed graph
    Returns:
        edges : m^2 x 3 array of edges: (point1, point2, type) [type is redundant knowing L, but still useful to have]
        L     : m^2 x 2 list of coordinates of the point enums in edges
    """
    L = np.arange(m**2).reshape((m,m))
    N = L.max()+1
    edges = []
    for i in range(0,m-1):
        edges += [(L[i,j], L[i+1,j],0) for j in range(m)]
    for i in range(0,m-1):
        edges += [(L[j,i], L[j,i+1],1) for j in range(m)]
    if periodic:
        edges += [(L[m-1,j], L[0,j],0) for j in range(m)]
        edges += [(L[j,m-1], L[j,0],1) for j in range(m)]
    edges = np.array(edges)
    edges = edges[np.argsort(edges[:,0])]
    # Coordinates 
    U = np.zeros((L.size,2))
    for i in range(m):
        for j in range(m):
            U[L[i,j]] = [i,j]
    U = U - m/2
    return edges, U

def star(vertex, edges):
    star = [j for j, e in enumerate(edges) if vertex in (e[0],e[1])]
    return list(star)

def plaq(vertex,edges):
    """
    Plaquette is identified with the specified `vertex` as its lower left corner
    In the picture below, v0 = vertex = first argument
       e2
    o------o 
    |      |
    |e3    | e1
    o------o
    v0  e0
    """
    A = np.arange(len(edges))
    e0 = A[(edges[:,0]==vertex) & (edges[:,2]==0)].item()
    e1 = A[(edges[:,0]==edges[e0,[1]] ) & (edges[:,2]==1)].item()
    e2 = A[(edges[:,1]==edges[e1,[1]] ) & (edges[:,2]==0)].item()
    e3 = A[(edges[:,0]==vertex) & (edges[:,2]==1)].item()
    plaq = [e0,e1,e2,e3]
    return plaq

def BasisOpX(basis, bit):
    basis = list(basis)
    basis[bit] = 0 if basis[bit] else 1
    return 1,tuple(basis)

def BasisOpZ(basis, bit):
    return Zphase[basis[bit]],basis

class State(object):
    def __init__(self,nqubits):
        self.nqubits = nqubits
        self.amplitude = []
        self.basisfunc = []   
    
    def add(self,amplitude, basisfunc):
        assert len(basisfunc)==self.nqubits
        basisfunc = tuple(basisfunc)
        found=False
        for i,b in enumerate(self.basisfunc):
            if b==basisfunc:
                self.amplitude[i] += complex(amplitude)
                found = True                
                break
        if not found:
            self.basisfunc += [basisfunc]
            self.amplitude = np.hstack((self.amplitude,[complex(amplitude)]))
        return self

    def normalize(self):
        A = np.sqrt((np.abs(self.amplitude)**2).sum())
        assert A>1.e-8
        self.amplitude /= A
        return self

    def OpX(self, ibit):
        for i in range(len(self.basisfunc)):
            b = list(self.basisfunc[i])
            b[ibit] = 0 if b[ibit] else 1
            self.basisfunc[i] = tuple(b)
        return self

    def OpY(self, ibit):
        for i in range(len(self.basisfunc)):
            b = list(self.basisfunc[i])
            b[ibit] = 0 if b[ibit] else 1
            self.basisfunc[i] = tuple(b)
            self.amplitude[i] *= complex(0,1)
        return self

    def OpZ(self, ibit):
        for i in range(len(self.basisfunc)):
            p = -1 if self.basisfunc[i][ibit] else 1
            self.amplitude[i] *= p
        return self

