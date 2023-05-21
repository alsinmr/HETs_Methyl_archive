#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:14:28 2022

@author: albertsmith
"""

import pyDR
import numpy as np
from pyDR.Fitting.fit import model_free
from pyDR.misc.tools import linear_ex
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from pyDR.misc.Averaging import avgData

frames=pyDR.Project('proj_md')
sub=frames['HETs_MET_4pw_SegB']['.+Met']


#%% Function to calculate librational amplitude from S2
""" We assume libration is dominated by small amplitude rotation around a fixed
angle of 109.49 degrees. Then, we calculate order parameters as a function of
a Gaussian distribution of rotation angles
"""

class S2_2_libA():    
    def __init__(self):
        """
        Calculates the amplitude (standard deviation) of librational motion 
        assuming a restricted wobbling-on-a-cone model with opening angle of
        109.5 degrees, where angles are distributed according to a Normal 
        distribution

        Parameters
        ----------
        S2 : float, array
            Order parameter(s) to calculate distribution for.

        Returns
        -------
        float, array
            Standard deviation of Gaussian distribution.

        """
        sigma0=np.atleast_2d(np.linspace(1,60,60))
        theta=np.atleast_2d(np.linspace(-180,180,201)).T
        P2=1/27*(32*np.cos(sigma0*np.pi/180)**2+8*np.cos(sigma0*np.pi/180)-13)
        prob=np.exp(-(theta**2)@(1/(2*sigma0**2)))
        prob[:,0]=0
        prob[100,0]=1
        prob/=prob.sum(0)
        
        self.S2=(prob*P2).sum(0)
        self.sigma=sigma0.squeeze()/np.sqrt(2)
    
    def __call__(self,S2):
        return linear_ex(self.S2,self.sigma,S2)
    
    def sigma2S2(self,sigma):
        return linear_ex(self.sigma,self.S2,sigma)

S2_2_libA=S2_2_libA()  #Initialize an instance to use as a function ("callable")


"Routine for fitting simulated methyl hopping/methyl libration data to a linear model"
class Z2lib():
    def __init__(self):
        """
        Initially finds a linear relationship between the log-correlation time (z)
        of methyl hopping to the amplitude (sigma), and then uses this relationship
        to convert between z (hopping) and sigma (libration)

        Returns
        -------
        None.

        """
        self.subproj=sub
        A=model_free(sub['MetLib'][0],nz=1,fixz=-14)[1][0]
        sigma=S2_2_libA(1-A)
        z=model_free(sub['MetHop'][0],nz=1,
                      fixA=[1-sub['MetHop'][0].R[:,7]])[0][0]
        
        M=np.concatenate(([np.ones(z.shape)],[z])).T
        
        index=np.ones(z.shape,dtype=bool)
        thresh=2
        for _ in range(2):
            self.b=np.linalg.lstsq(M[index],np.atleast_2d(1/(sigma[index]**2)).T)[0]
            self.error=self.b[0]+self.b[1]*z-1/(sigma**2)
            self.stdev=np.sqrt((self.error[index]**2).sum()/(index.sum()-2))
            if not(np.any(np.abs(self.error[index])>thresh*self.stdev)):
                break
            index=np.abs(self.error)<thresh*self.stdev
        
        self.index=index
        self.A=A
        self.S2=1-A
        self.sigma=sigma
        self.z=z
        self.sens=None
        self.rho=None
        
    def z2lib(self,z):
        """
        Finds sigma of the libration as a function of the log-correlation time of
        methyl hopping

        Parameters
        ----------
        z : float, array
            Log10 of the methyl hopping correlation time.

        Returns
        -------
        None.

        """
        x=self.b[0]+self.b[1]*z
        index=x<=0
        x[index]=1e-12
        sigma=x**(-1/2)
        sigma[sigma>S2_2_libA(0)]=S2_2_libA(0) #restrict so that it doesn't go over S2=0
        return sigma
    
    def z2S2(self,z):
        """
        Estimates the order parameter for methyl libration (S2) from the 
        log-correlation time (z).

        Parameters
        ----------
        z : list-like
            List of log-correlation times used to calculate S2.

        Returns
        -------
        S2
            Order parameters for librational motion.

        """
        return S2_2_libA.sigma2S2(self.z2lib(z))
    
    def plot(self):
        """
        Plots the fit of correlation time of hopping to order parameter of 
        librational motion. Shown as two plots, one with log-correlation time
        vs. S2, and the other with log-correlation time  vs. 1/sigma^2, the
        latter being the linear plot. Data points excluded from the linear
        fit are shown in red.

        Returns
        -------
        None.

        """
        fig=plt.figure()
        ax=[fig.add_subplot(1,2,k+1) for k in range(2)]
        ax[0].scatter(self.z[self.index],self.sigma[self.index])
        ni=np.logical_not(self.index)
        ax[0].scatter(self.z[ni],self.sigma[ni],color='red',marker='x')
        z=np.sort(self.z)
        ax[0].plot(z,self.z2lib(z),color='black')
        # ax[0].set_xlabel(r'log$_{10}$($\tau_c$/s)')
        ax[0].set_xticks(np.linspace(-11,-10.2,5))
        ax[0].set_xticklabels([None if k%2 else '{:.0f} ps'.format(10**v*1e12) for k,v in enumerate(ax[0].get_xticks())])
        ax[0].set_xlabel(r'$\tau_c$')
        ax[0].set_ylabel(r'$\sigma$ / $^\circ$')
        
        ax[1].scatter(self.z[self.index],1/self.sigma[self.index]**2)
        ax[1].scatter(self.z[ni],1/self.sigma[ni]**2,color='red',marker='x')
        ax[1].plot(z,self.b[0]+self.b[1]*z,color='black')
        ax[1].set_xticks(np.linspace(-11,-10.2,5))
        ax[1].set_xticklabels([None if k%2 else '{:.0f} ps'.format(10**v*1e12) for k,v in enumerate(ax[0].get_xticks())])
        ax[1].set_xlabel(r'$\tau_c$')
        ax[1].set_ylabel(r'$1/\sigma^2 / (^\circ)^{-2}$')
        fig.set_size_inches([8.5,4])
        fig.tight_layout()
        
    def plot_z_v_S2(self):
        """
        Plots the fit of correlation time of hopping to order parameter of 
        librational motion. Shown as two plots, one with log-correlation time
        vs. S2, and the other with log-correlation time  vs. 1/sigma^2, the
        latter being the linear plot. Data points excluded from the linear
        fit are shown in red.

        Returns
        -------
        None.

        """
        fig=plt.figure()
        ax=[fig.add_subplot(1,2,k+1) for k in range(2)]
        ax[0].scatter(self.z[self.index],self.S2[self.index])
        ni=np.logical_not(self.index)
        ax[0].scatter(self.z[ni],self.S2[ni],color='red',marker='x')
        z=np.sort(self.z)
        ax[0].plot(z,self.z2S2(z),color='black')
        # ax[0].set_xlabel(r'log$_{10}$($\tau_c$/s)')
        ax[0].set_xticks(np.linspace(-11,-10.2,5))
        ax[0].set_xticklabels([None if k%2 else '{:.0f} ps'.format(10**v*1e12) for k,v in enumerate(ax[0].get_xticks())])
        ax[0].set_xlabel(r'$\tau_c$')
        ax[0].set_ylabel(r'$S^2$')
        
        ax[1].scatter(self.z[self.index],1/self.sigma[self.index]**2)
        ax[1].scatter(self.z[ni],1/self.sigma[ni]**2,color='red',marker='x')
        ax[1].plot(z,self.b[0]+self.b[1]*z,color='black')
        ax[1].set_xticks(np.linspace(-11,-10.2,5))
        ax[1].set_xticklabels([None if k%2 else '{:.0f} ps'.format(10**v*1e12) for k,v in enumerate(ax[0].get_xticks())])
        ax[1].set_xlabel(r'$\tau_c$')
        ax[1].set_ylabel(r'$1/\sigma^2 / (^\circ)^{-2}$')
        fig.set_size_inches([8.5,4])
        fig.tight_layout()
        
    def plot_MFvDet(self):
        proj=self.subproj._parent
        proj.current_plot=1
        self.subproj['MetLib'].plot(style='bar')
        proj['ModelFreeFit']['MetLib'].plot()
        for a in proj.plot_obj.ax[:-1]:
            a.set_ylim([0,self.subproj['MetLib'].R[:,:-1].max()*1.1])
        proj.current_plot=2
        self.subproj['MetHop'].plot(style='bar')
        proj['ModelFreeFit']['MetHop'].plot()
        for a in proj.plot_obj.ax[:-1]:
            a.set_ylim([0,self.subproj['MetHop'].R[:,:-1].max()*1.1])
        
    def fit2z_sigma(self,data,n:int=2):
        """
        Uses the first n detectors of a data object and fits these to a model 
        with a correlation time of hopping (z) and amplitude of libration (sigma),
        assuming these two parameters are dependent.

        Parameters
        ----------
        data : pyDR data object
            Data object to fit.
        n : int, optional
            Number of detectors to use. The default is 2.

        Returns
        -------
        None.

        """
        self.fit_setup(data.sens)
        z=list()
        z0=list()
        for rho in data.R[:,:n]:
            i=np.argmin(((self.rho[:n].T-rho)**2).sum(1))
            def fun(z):
                rhoc=linear_ex(self.sens.z,self.rho[:n],z).squeeze()
                return (rhoc-rho)
            out=leastsq(fun,self.sens.z[i])
            z.append(out[0][0])
            
            i=np.argmin(((self.sens.rhoz[:n].T-rho)**2).sum(1))
            def fun(z):
                rhoc=linear_ex(self.sens.z,self.sens.rhoz[:n],z).squeeze()
                return (rhoc-rho)
            out=leastsq(fun,self.sens.z[i])
            z0.append(out[0][0])
        
        z,z0=np.array(z),np.array(z0)
        sigma=self.z2lib(z)
        return z,sigma,z0
            
            
    
    def fit_setup(self,sens):
        """
        Calculates a matrix of detector responses as a function of the methyl
        hopping correlation time (z)

        Parameters
        ----------
        sens : sens object
            pyDR sensitivity object.

        Returns
        -------
        None.

        """
        if self.sens is not None and sens==self.sens:return

        z=sens.z
        # sigma=self.z2lib(z)
        # S2=S2_2_libA.sigma2S2(sigma)
        S2=self.z2S2(z)
        
        self.rho=sens.rhoz*(8/9-1+S2)+\
            np.atleast_2d(sens.rhoz[:,0]).T@np.atleast_2d((1-S2))
        self.sens=sens        





