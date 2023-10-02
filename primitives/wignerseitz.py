"""
Generate the nearest neighbors of gamma point
Define the Bragg plane for each neighbor
Compute the mid points to next nearest neighbors
For each mid point, retain those inside the Bragg Volume
Using new points, place new Bragg planes and update the Bragg volume
Continue to next neighbors until no mid points found inside a Bragg volume

We need: 
1. A function that generates nearest neighbors of a given order
2. A way to describe closed Bragg volume
3. A function that tests if a point lies inside Bragg volume

(3) is easy: a point lies inside Bragg volume if it is on the side of gamma point of every Bragg plane
A point, 'p' lies on the same side as the Gamma point of a plane (n, d) if (p-d).n < 0
"""
import numpy as np
from itertools import product
try:
    import vtk
    import pyvista
except:
    pass

getint = lambda A : np.linalg.solve(A, (A**2).sum(axis=1))
inplane = lambda r, p : np.abs((p*(r-p)).sum())<1.e-12
unitvec = lambda v : v/np.sqrt((v**2).sum())

def lattice_points_nnorder(a1,a2,a3=None,maxorder=3):
    n1 = np.arange(-maxorder, maxorder+1)
    if a3==None:
        a = np.vstack((a1,a2))
    else:
        a = np.vstack((a1,a2,a3))
    L = np.sqrt((a**2).sum(axis=1))
    maxL = np.max(L)
    imax = np.argmax(L)
    M = np.array([maxorder,maxorder,maxorder])+1
    maxd = maxL * maxorder
    M = [int(maxL/l)*maxorder for l in L]
    M = [np.arange(-m,m+1) for m in M]
    nmk = np.meshgrid(*M)
    nmk = np.vstack([c.flatten() for c in nmk]).T
    if a3 is None:
        R = nmk[:,:2] @ np.vstack((a1,a2))
    else:
        R = nmk @ np.vstack((a1,a2,a3))
    d = (R**2).sum(axis=1)
    iord = np.argsort(d)
    R = R[iord]
    d = d[iord]
    nmk = nmk[iord]

    du,di = np.unique(d,return_index=True)
    n = di[maxorder+2]
    R = R[:n]
    nmk = nmk[:n]
    return R
#
def same_side(point, lattice_vector):
    p = np.asarray(point).flatten()
    R = np.asarray(lattice_vector).flatten()
    r = np.sqrt((R*R).sum())
    assert ( r > 1.e-12 )
    R = R/r
    p = (p*R).sum()
    return p < r
#
def wigner_seitz_planes(a1,a2,a3,maxorder=6):
    """
    Returns the surface planes in terms of the shortest vector from origin to the Bragg plane
    """
    R = lattice_points_nnorder(a1,a2,a3,maxorder=maxorder)
    S = R.copy()
    B = [1]
    for j in range(2,len(R)):
        # Keep only the points on the same side of all planes in B
        S = []
        for i in range(len(R)):
            keep = all([same_side(R[i]/2*1.001,R[k]/2) and i != k for k in B])
            if keep:
                S.append(i)
        S = np.sort(S)
        # Now add another Bragg Plane, and eliminate more points from S
        S2 = np.sort([i for i in S if same_side(R[i]/2*1.001,R[j]/2) and i != j])
        if len(S2)==len(S):
            if np.all(S==S2):
                break
        # Else add jth point for Bragg plane
        B.append(j)
    B = R[B]/2
    return B
#
def wigner_seitz_intersections2d(B):
    """
    Let B = [b1,b2] be the vector from the origin intersecting the line, i.e. shortest distance vector from origin
        C = [c1,c2] for the second line
    Intersection:

    1/|B| (B . (r - B)) = 0
    1/|C| (C . (r - C)) = 0
    
    B.r = B.B
    C.r = C.C

    [b1, b2][x]    [B.B]
    [c1, c2][y]  = [C.C]

    """
    assert (B.shape[1]==2)
    Bhat = B / np.sqrt((B**2).sum(axis=1))[:,None]
    P = []
    for i in range(1,len(B)):
        for j in range(0,i):
            A = np.array([B[i],B[j]])
            if np.abs(np.linalg.det(A))>1.e-12:
                xy = np.linalg.solve( A, [ B[i] @ B[i], B[j] @ B[j]  ] )
                if all([same_side(xy*(1-1e-3), b) for b in B]):
                    P.append(xy)
    P = np.vstack(P)
    t = np.arctan2(P[:,1],P[:,0])
    i = np.argsort(t)
    P = P[i]
    return P
#

def duplicates(P,thresh=1e-3):
    Q = []
    for m in range(0, len(P)-1):
        d = np.abs(P[(m+1):]-P[m]).max(axis=1)
        i = np.argmin(d)
        if d[i] < thresh:
            Q.append((i + m + 1,m))
    Q = np.vstack(Q)
    return Q  

def wigner_seitz_intersections3d(B):
    """
    Let B = [b1,b2,b3] be the vector from the origin intersecting the plane, i.e. shortest distance vector from origin
        C = [c1,c2,c3] for the second line
        D = [d1,d2,d3] for the second plane
    Intersection:
    (B . (r - B)) = 0
    (C . (r - C)) = 0
    (D . (r - D)) = 0
    """
    assert (B.shape[1]==3)
    Bhat = B / np.sqrt((B**2).sum(axis=1))[:,None]
    Surfaces = []
    IntPoints = []
    n = len(B)
    IJK = np.vstack(list(product(range(n),range(n),range(n))))
    IJK = np.unique(np.sort(IJK,axis=-1),axis=0)

    for m,(i,j,k) in enumerate(IJK):
        A = np.array((B[i],B[j],B[k]))
        a = np.abs(np.linalg.det(A))
        if a > 1.e-12:
            r = getint(A)
            interior = all([same_side(r*0.99, b) for b in B])
            if interior:
                if IntPoints:
                    d = np.abs(np.array(IntPoints)-r).max(axis=1)
                    q = np.argmin(d)
                    keep = d[q] > 1.e-6
                else:
                    keep = True
                if keep:
                    IntPoints.append(r)
                    Surfaces.append( [i,j,k,len(IntPoints)-1] ) 
                else:
                    Surfaces.append( [i,j,k,q] )
    P = np.vstack(IntPoints)
    Surfaces = np.vstack(Surfaces)

    # Now arrange by midpoints
    uniq = np.unique(Surfaces[:,:3])
    unitvec = lambda v : v/np.sqrt((v**2).sum())
    Segs = []
    for m in uniq:
        sel = np.any(Surfaces[:,:3]==m,axis=1)
        corners = np.unique(Surfaces[sel][:,-1])
        # basis vectors
        u1 = unitvec(P[corners[0]]-B[m])
        u2 = unitvec(np.cross( u1, B[m]))
        vecs = np.vstack([P[c]-B[m] for c in corners])
        local_coords = np.vstack([ [v @ u1, v @ u2] for v in vecs ])
        check = np.all( np.abs(((vecs**2).sum(axis=1) - (local_coords**2).sum(axis=1))) < 1.e-12 )
        assert (check)
        angles = np.arctan2(local_coords[:,1],local_coords[:,0])
        isort = np.argsort(angles)
        Segs.append((m, corners[isort]))
    return P,Segs

def triangulate_wigner_seitz(BraggPts,Corners,Surfaces):
    xyz = np.vstack((BraggPts,Corners))
    n = len(BraggPts)
    triangles=[]
    for m, c in Surfaces:
        for i in range(len(c)):
            t = [ m, n + c[i], n + c[(i+1)%len(c)] ]
            triangles.append(t)
    return xyz,triangles

def polydata_wigner_seitz(Corners,Surfaces):
    polygons = []
    for m, c in Surfaces:
        p = [len(c)] + list(c)
        polygons.append(p)
    return polygons

def plot_wigner_seitz(BraggPts,Corners,Surfaces,opacity=1.0):
    faces = polydata_wigner_seitz(Corners,Surfaces)
    faces = np.hstack(faces)
    surf = pyvista.PolyData(Corners,faces)
    p = pyvista.Plotter()
    p.add_mesh(surf,show_edges=True,opacity=opacity)
    p.add_points(BraggPts,render_points_as_spheres=True,point_size=12,color='y')
    p.add_points(Corners,render_points_as_spheres=True,point_size=10,color='c')
    p.show()

def transform_table(M,B,thresh=1e-6):
    Bt = B @ M
    T=[]
    for j,b in enumerate(Bt):
        d = np.sqrt( ((B - b)**2).sum(axis = 1) )
        i = np.argmin(d)
        T.append(i)
    if np.abs(B[T]-Bt).max() > thresh:
        return None
    return np.array(T)