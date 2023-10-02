"""
Mean-Field Approximate Optimization Algorithm
Aditi Misra-Spieldenner1, Tim Bode2,3, Peter K. Schuhmacher3,
Tobias Stollenwerk2,3, Dmitry Bagrets2,4, and Frank K. Wilhelm1,2

arXiv:2303.00329v1 [quant-ph] 1 Mar 2023
"""
import numpy as np
from math import sqrt
from scipy import stats
from scipy import optimize
from argparse import ArgumentParser

class MeanFieldSK(object):
    def __init__(self,N=200,Jsq=1.0,tau=0.5,p=12,seed=None):
        self.N = N
        self.VD_matrices = np.zeros((N,3,3))
        for i in range(3):
            self.VD_matrices[:,i,i] = 1
        self.VP_matrices = np.zeros((N,3,3))
        for i in range(3):
            self.VP_matrices[:,i,i] = 1
        self.nvec = np.zeros((N,3))
        self.nvec[:,0] = 1
        self.magz_eff = np.zeros(N)
        rng = np.random.default_rng(seed)
        self.J = rng.normal(scale=sqrt(Jsq/N),size=(N,N))
        self.J    = np.triu(self.J,k=1)
        self.J = (self.J + self.J.T)/2
        self.J = self.J - np.diag(np.diag(self.J))        
        # self.triuJ = np.triu(np.ones(self.J.shape,dtype=bool),k=1)
        # self.Jvar = self.J[self.triuJ].var()
        # To remove degeneracy due to Z2 symmetry
        self.h    =self.J[:,-1]
        self.h[-1] = 0.
        self.Delt = np.ones(N)
        self.cost()
        self.p    = p
        self.tau  = tau

    def sigma_vec(self):
        return np.sign(self.nvec)

    def update_param(self,name,val):
        if hasattr(self,name):
            setattr(self,name,val)
        return self

    def update_magz_eff(self):
        self.magz_eff = self.h + self.J @ self.nvec[:,2]
        return self
    
    def cost(self):
        M = self.update_magz_eff()
        H = -sum( self.magz_eff * self.nvec[:,2] )
        self.energy = H
        return H 

    def cost_spin(self):
        s = np.sign(self.nvec[:,2])
        return -(s @ (self.J @ s))/self.N

    def propagate_state(self,tau=None,p=None,verbose=0):
        self.initialize_state()
        if tau is not None:
            self.tau = tau
        if p is not None:
            self.p = p
        for k in range(1,self.p):
            self.time_step(k)
        if verbose:
            print('Num time steps = ',k+1)
        return self
    
    def initialize_state(self):
        self.nvec[:] = 0
        self.nvec[:,0] = 1
        return

    def time_step(self,k=1):
        # Compute magnetization vector
        self.update_magz_eff()
        # Compute the VD matrices
        self.compute_VD(k)
        # Compute the VP matrices
        self.compute_VP(k)
        # Apply the iteration step
        self.nvec = np.einsum('ijk,ik->ij',self.VP_matrices,self.nvec)
        self.nvec = np.einsum('ijk,ik->ij',self.VD_matrices,self.nvec)
        return self

    def compute_VD(self,k):
        beta_k = self.tau*(1-(k-1)/self.p)
        w = 2*self.Delt*beta_k
        self.VD_matrices[:,1,1] =  np.cos(w)
        self.VD_matrices[:,2,2] =  np.cos(w)
        self.VD_matrices[:,1,2] =  np.sin(w)
        self.VD_matrices[:,2,1] = -np.sin(w)
        return self

    def compute_VP(self,k):
        gamma_k = self.tau/self.p * k
        w       = 2*gamma_k*self.magz_eff
        self.VP_matrices[:,0,0] =  np.cos(w)
        self.VP_matrices[:,1,1] =  np.cos(w)
        self.VP_matrices[:,0,1] =  np.sin(w)
        self.VP_matrices[:,1,0] = -np.sin(w)
        return self

    def costfunc(self,var):
        tau, p = var
        E = self.propagate_state(tau,p).cost()
        return E

def main(args):
    obj = MeanFieldSK(N=args.N, Jsq = args.Jsq, tau=0.5, p=10000)
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('-N',type=int,default=200)
    parser.add_argument('-Jsq',type=int,default=1.0)
    args=parser.parse_args()
    main(args)


    
    
    
    