#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:25:02 2022

@author: albertsmith
"""

import numpy as np
import pyDR
import matplotlib.pyplot as plt

vft=pyDR.MDtools.vft
linear_ex=pyDR.tools.linear_ex



class S_v_pop():
    def __init__(self):
        """
        Upon initialization, we will pre-calculate S as a function of p1, assuming
        that p2=p3 (or p3=0?) Does it matter? 
        
        When the class is called with S as input, we will then back-calculate and
        return p1

        Returns
        -------
        None.

        """
        
        
        #Tensors for each of three states
        theta=np.arccos(np.sqrt(1/3))*2 #Tetrahedral angle
        A0=vft.D2(0,[theta,theta,theta],[0,2*np.pi/3,4*np.pi/3])
        
        #Solve for p3=0
        self.p1=np.linspace(0,1,101)
        self.p2,self.p3=1-self.p1,np.zeros(self.p1.shape)
        
        A=np.array([sum([p*a for p,a in zip([self.p1,self.p2,self.p3],A)]) for A in A0])  #Residual tensor
        
        delta=vft.Spher2pars(A)[0]
        self.S=delta*np.sqrt(3/2) #Order parameter

        #Solve for p2=p3
        self.p2eq=self.p3eq=(1-self.p1)/2
        
        A=np.array([sum([p*a for p,a in zip([self.p1,self.p2eq,self.p3eq],A)]) for A in A0])  #Residual tensor
        
        delta=vft.Spher2pars(A)[0]
        self.Seq=delta*np.sqrt(3/2) #Order parameter
        
    def plot(self):
        """
        Plots S as a function of p1, assuming both p3=0 and p2=p3

        Returns
        -------
        None.

        """
        ax=plt.figure().add_subplot(111)
        
        ax.plot(self.p1,self.S,color='red')
        ax.plot(self.p1,self.Seq,color='blue')
        
        ax.plot(self.p1,np.abs(self.S),color='red',linestyle=':')
        ax.plot(self.p1,np.abs(self.Seq),color='blue',linestyle=':')
        
        ax.legend((r'$p_3$=0',r'$p_2=p_3$'))
        ax.set_xlabel(r'$p_1$')
        ax.set_ylabel(r'S')
        
    def get_pop(self,S):
        """
        Returns p1 as a function of S. Returns values of p1 first assuming p3=0,
        second assuming p2=p3 where p1>p2,p3, and third assuming p2=p3 where 
        p1<p2,p3 (that is, p1>1/3 and p1<1/3). 
        
        Note that if S<1/3 (which would be impossible for tetrahedral hops), then
        the best solution is p1=p2=p3=1/3. Since the first entry assumes p3=0, 
        then this will return NaN, whereas the second and third outputs will
        yield p1=1/3. S>1 will yield p1=1 for all outputs

        Parameters
        ----------
        S : float or list-like
            Order parameter.

        Returns
        -------
        np.array
            p1 assuming
                1) p3=0
                2) p2=p3, with p1>=p2,p3
                3) p2=3, with p1<=p2,p3

        """
        S=np.abs(np.atleast_1d(S))  #Fit absolute values
        
        i=len(self.S)>>1
        p1=linear_ex(self.S[i:],self.p1[i:],S)
        i=np.argmin(np.abs(self.Seq))
        p1eq1=linear_ex(np.abs(self.Seq[i:]),self.p1[i:],S)
        p1eq2=linear_ex(np.abs(self.Seq[:i]),self.p1[:i],S)
        
        #Disallowed values
        i=S<1/3
        p1[i]=np.nan
        p1eq1[i]=1/3
        p1eq2[i]=1/3
        
        i=S>1
        p1[i]=1
        p1eq1[i]=1
        p1eq2[i]=1
        
        i=S>-self.Seq[0]
        p1eq2[i]=np.nan
        
        i=S<0.5
        p1[i]=np.nan
        
        return np.concatenate(([p1],[p1eq1],[p1eq2]),axis=0).squeeze()
    
    def __call__(self,S,resid):
        """
        Returns a single population for a given S and resid(s) (only one S allowed)

        Parameters
        ----------
        S : TYPE
            DESCRIPTION.
        resid : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        which=self.which_pop(resid)
        if hasattr(which,'__len__'):which=which[0]
        
        
        if S<.5 and which==0:which=1
        pop=self.get_pop(S)[which]
        
        if which==0:
            return np.sort([pop,1-pop,0])[::-1]
        return np.sort([pop,(1-pop)/2,(1-pop)/2])[::-1]
        
        
        
        
    
    def which_pop(self,resid):
        """
        Determines which assumptions to take on populations based on the results of
        populations from the MD trajectory (HETs_MET_4pw)

        Parameters
        ----------
        resid : int
            Which resid to use.

        Returns
        -------
        int
            0 : assume p3=0
            1 : assume p2=p3, p1>=1/3
            2 : assume p2=p3, p1<=1/3
        """
        if not(hasattr(self,'md')):self.md=load_md()[1]
        
        if hasattr(resid,'__len__'):
            return [self.which_pop(res) for res in resid]
        
        pop=self.md[resid]['pop_outer']  #Get the outer population
        
        if pop[0]>1/3 and pop[2]>0.5*pop[1]:return 1  #pop[2] a significant fraction of pop[1]
        if pop[0]>0.5:return 0
        if pop[1]/pop[0]>pop[2]/pop[1]:return 2  #First two populations equal, large compared to third (third pop is p1)
        return 1
        
              
S_v_pop=S_v_pop()  #Initialize the class (use as function, i.e. callable)


def load_md():
    """
    Loads in the populations from the MD trajectories, that were extracted
    with the script extract_chi_pops.py

    Returns
    -------
    Tuple of dictionaries
        First dictionary corresponds to the trajectory HETs_4pw, the second
        dictionary corresponds to the trajectory HETs_MET_4pw. The former
        uses 4-point water without methyl correction, and the latter 4-point
        water with methyl correction.

    """
    md_no=dict()
    md_corr=dict()
    for file,md in zip(['chi12_populations/chi12_HETs_4pw.txt','chi12_populations/chi12_HETs_MET_4pw_SegB.txt'],[md_no,md_corr]):
        with open(file,'r') as f:
            for k,l in enumerate(f):
                if k==0:continue
                
                resid,resname,pop_outer,pop_inner=l.strip().split()
                if resname=='ALA':continue  #no data for ala
                resid=int(resid)
                pop_outer=np.array([float(p) for p in pop_outer[1:-2].split(',')])
                
                # if resid in md:md[resid+0.1]=md[resid]
                
                md[resid]={'Name':resname,'pop_outer':np.sort(pop_outer)[::-1]}
                
                if resname in ['LEU','ILE']:
                    pop_inner=np.array([float(p) for p in pop_inner[1:-1].split(',')])
                    md[resid]['pop_inner']=np.sort(pop_inner)[::-1]
                    
    return md_no,md_corr


if __name__=='__main__':
    #Read out MD population results
    
    md_no,md_corr=load_md()
    
    #Plot experiment and simulation                
    
    
    #Read in NMR results and residues
    data=pyDR.Project('directHC')['NMR']['raw']
    S=data.S2**(0.5)*3  #Remove methyl contribution
    resids=[int(lbl[:3]) if len(lbl.split(','))==1 else [int(l0[:3]) for l0 in lbl.split(',')] \
            for lbl in data.label]   
    names=[lbl[3:].upper() if len(lbl.split(','))==1 else 'VAL' for lbl in data.label]
    
    w=0.13
    w1=0.25
    off=.03
    xlabel=list()
    xlabel1=list()
    
    ax=plt.figure().add_subplot(111)
    ax1=plt.figure().add_subplot(111)
    clrs=(plt.get_cmap('tab10')(7),plt.get_cmap('tab10')(0),plt.get_cmap('tab10')(1))
    k=-1
    pop=list()
    for (resid,S0) in zip(resids,S):
        if not(hasattr(resid,'__len__')) and resid not in md_no:
            pop.append(1)
            continue
        
        pop0=S_v_pop(S0,resid)
        pop.append(pop0[0])
        k+=1
        
        if hasattr(resid,'__len__'):
            po=np.mean([md_no[res]['pop_outer'] for res in resid],axis=0)
            poc=np.mean([md_corr[res]['pop_outer'] for res in resid],axis=0)
            pi=None
            pic=None
        else:
            po,poc=md_no[resid]['pop_outer'],md_corr[resid]['pop_outer']
            pi,pic=(md_no[resid]['pop_inner'],md_corr[resid]['pop_inner']) if 'pop_inner' in md_no[resid] else (None,None)
        

        if pi is None:
            ax.bar(k-w,pop[-1],width=w,color=clrs[0],edgecolor='black')
            ax.bar(k,po[0],width=w,color=clrs[1],edgecolor='black')
            ax.bar(k+w,poc[0],width=w,color=clrs[2],edgecolor='black')
        else:
            ax.bar(k-w*2.5-off,1,width=w,color=clrs[0],edgecolor='black')
            ax.bar(k-w*1.5-off,pi[0],width=w,color=clrs[1],edgecolor='black')
            ax.bar(k-w*0.5-off,pic[0],width=w,color=clrs[2],edgecolor='black')
            ax.bar(k+w*0.5+off,pop[-1],width=w,color=clrs[0],edgecolor='black')
            ax.bar(k+w*1.5+off,po[0],width=w,color=clrs[1],edgecolor='black')
            ax.bar(k+w*2.5+off,poc[0],width=w,color=clrs[2],edgecolor='black')
        
        pop1=S_v_pop.get_pop(S0)
        ax1.bar(k-w1,pop1[0],width=w1,color=clrs[0],edgecolor='black')
        ax1.bar(k,pop1[1],width=w1,color=clrs[1],edgecolor='black')
        ax1.bar(k+w1,(1-pop1[2])/2,width=w1,color=clrs[2],edgecolor='black')

        string=r'$p_3=0$' if pop0[-1]==0 else (r'$p_2=p_3$' if pop0[1]==pop0[2] else r'$p_1=p_2$')
        ax.text(k,1.1,string,horizontalalignment='center')
            
        if hasattr(resid,'__len__'):
            xlabel.append(r'$\chi_1$'+'\n'+',\n'.join([f'{res}Val' for res in resid]))
            xlabel1.append(xlabel[-1])
        else:
            if pi is None:
                xlabel.append(r'$\chi_1$'+'\n'+f'{resid}{md_no[resid]["Name"]}')
                xlabel1.append(xlabel[-1])
            else:
                xlabel.append(r'$\chi_1$ $\chi_2$'+'\n'+f'{resid}{md_no[resid]["Name"]}')
                xlabel1.append(r'$\chi_2$'+'\n'+f'{resid}{md_no[resid]["Name"]}')
            
    ax.set_xticks(range(len(xlabel)))
    ax.set_xticklabels(xlabel)
    ax.legend(('NMR (S)','MD (no corr.)','MD (met. corr.)'),loc='lower left')
    ax.set_ylabel('major population')
    ax.figure.set_size_inches([6.8,4.8])
    ax.figure.tight_layout()
    ax.set_yticks(np.linspace(0,1,5))
    
    ax1.set_xticks(range(len(xlabel)))
    ax1.set_xticklabels(xlabel1)
    ax1.legend((r'$p_3=0$',r'$p_2=p_3$',r'$p_1=p_2$'),loc='lower left')
    ax1.set_ylabel('major populaion')
    ax1.figure.set_size_inches([6.8,4.8])
    ax1.figure.tight_layout()
    ax1.set_yticks(np.linspace(0,1,5))
    
    data.project.chimera.close()    
    data.select.chimera(x=1-np.array(pop),norm=False,color=(.1,.1,.1,1))    
    resi=np.concatenate([[str(res0) for res0 in resid] if hasattr(resid,'__len__') else [str(resid)] for resid in resids])
    data.project.chimera.command_line(['~show','show /B:'+','.join(resi)+'&:ILE,LEU,VAL',
                                       '~ribbon','ribbon /B','lighting soft','graphics silhouettes true'])
    
            
                    
        
    #Make a plot where we compare results of assumping p=3, p2=p3, p1=p2

        
                    
                
                
                
                
                
                
                
                
                
                