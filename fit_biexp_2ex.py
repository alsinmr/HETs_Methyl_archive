#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:26:15 2022

@author: albertsmith
"""

import numpy as np
from scipy.optimize import least_squares
from copy import copy
import pyDR
linear_ex=pyDR.tools.linear_ex

class Biexp2ex():
    def __init__(self,pop=[1,1,1]):
        """
        Class used for extracting the exchange matrix for 3-site tetrahedral
        hopping from amplitudes and correlation times obtained from frame 
        analysis. We start just with the populations of the three sites, which
        allows us to 
        """
        
        self.pop=np.array(pop,dtype=float)
        
        
        
    def __setattr__(self,name,value):
        if name=='pop':
            value[value==0]=1e-12  #No zeros
            value/=value.sum()  #Check normalize
        super.__setattr__(self,name,value)
            
            
        
    @property
    def S2(self):
        """
        Calculates the order parameter, S2, from the 3 populations

        Returns
        -------
        float
            S2.

        """
        pop=self.pop
        return (pop**2).sum()-2/3*(pop[0]*pop[1]+pop[0]*pop[2]+pop[1]*pop[2])
    
    def kex(self,k01,k02,k12):
        """
        Calculates the 3x3 exchange matrix assuming detailed balance from k12,
        k13, and k23

        Parameters
        ----------
        k01 : float
            0-1 rate constant
        k02 : float
            0-2 rate constant
        k12 : float
            1-2 rate constant

        Returns
        -------
        np.array
            3x3 exchange matrix

        """
        
        pop=self.pop
        kex=np.zeros([3,3])
        k10,k20,k21=k01*pop[0]/pop[1],k02*pop[0]/pop[2],k12*pop[1]/pop[2]
        
        kex=np.array([[-(k01+k02), k10,      k20],
                      [ k01,      -(k10+k12),k21],
                      [ k02,       k12,      -(k20+k21)]])
        
        return kex
    
    def k2tcA(self,k01,k02,k12):
        """
        Calculates parameters tc and A from the exchange matrix.

        Parameters
        ----------
        k01 : float
            0-1 rate constant
        k02 : float
            0-2 rate constant
        k12 : float
            1-2 rate constant

        Returns
        -------
        tuple
        z=log10(tc) (2 elements), A (2 elements)

        """
        
        w,v=np.linalg.eig(self.kex(k01,k02,k12))
        
            
        i=np.argsort(w)[::-1]
        w=w[i]
        v=v[:,i]
        peq=v[:,0].real/v[:,0].real.sum() #Equilibrium, normalized to sum to one
        
        vi=np.linalg.pinv(v)
 
        P2=-1/3*np.ones([3,3])+4/3*np.eye(3)
        pp=np.dot(np.atleast_2d(peq).T,np.atleast_2d(peq))
        
        S2=(P2*pp).sum()    #Order parameter
        
        tc=-1/w[1:].real    #Correlation times
        
        A=list()
        for vim,vm in zip(vi,v.T):
            A.append((np.dot(np.atleast_2d(vm).T,np.atleast_2d(vim*peq))*P2).sum())
        A=np.array(A).real
        
        A=A[1:]
        
        return np.log10(tc),A
    
    def fit(self,pop,data):
        sens=data.sens
        Rstd=data.Rstd[0]
        
        k=list()
        Rc=list()
        
        for p,R in zip(pop.T,data.R):
            self.pop=p
            def calc_rho(k01,k02,k12):
                z,A=self.k2tcA(k01,k02,k12)
                try:
                    return linear_ex(sens.z,sens.rhoz,z[0])*A[0]+\
                              linear_ex(sens.z,sens.rhoz,z[1])*A[1]+\
                              linear_ex(sens.z,sens.rhoz,sens.z[-1])*self.S2
                except:
                    print(k01,k02,k12)
                          
            def fit_fun(logk):
                k=10**logk
                return (calc_rho(k[0],k[1],k[2])-R)/Rstd

            out=least_squares(fit_fun,[7,8,9])                
            k.append(10**out.x)
            Rc.append(calc_rho(*k[-1]))
                
        out=copy(data)
        out.R=np.array(Rc)
        
        out.source.details.append('Back calculation of detector responses from 3-site exchange model')
        out.source.Type='ExchangeFit'
        out.source.src_data=None
        
        if data.source.project is not None:data.source.project.append_data(out)
        
        return np.array(k),out
        
    
    