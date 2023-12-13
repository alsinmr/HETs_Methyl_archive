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
        theta=np.arccos(-1/3) #Tetrahedral angle
        A0=vft.D2(0,[theta,theta,theta],[0,2*np.pi/3,4*np.pi/3])
        
        #Solve for p3=0
        self.p1=np.linspace(0,1,301)
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
        ax.vlines(1/3,-.5,1,color='grey',linestyle=':')
        
        cmap=plt.get_cmap('tab10')
        half=len(self.p1)//2
        third=len(self.p1)//3
        
        ax.fill_between(self.p1[half:],self.S[half:],self.Seq[half:],color='lightgrey')
        ax.fill_between(self.p1[third:half+1],np.abs(self.Seq[third:half+1]),
                        np.abs(self.Seq[third+1:0:-2]),color='lightgrey')
        
        hdls=[]
        hdls.append(ax.plot(self.p1[half:],self.S[half:],color=cmap(0))[0])
        hdls.append(ax.plot(self.p1[third:half],self.Seq[third:half],color=cmap(1))[0])
        hdls.append(ax.plot(self.p2eq[1:third],self.Seq[1:third],color=cmap(6))[0])
        ax.plot(self.p1[half:],self.Seq[half:],color=cmap(1))
        ax.plot(self.p1[third:],np.abs(self.Seq[third:]),color=cmap(1),linestyle=':')
        ax.plot(self.p2eq[:third],np.abs(self.Seq[:third]),color=cmap(6),linestyle=':')
        
        ax.legend(hdls,(r'$p_1>p_2,p_3=0$',r'$p_1>p_2=p_3$',r'$p_1=p_2>p_3$'))
        ax.set_xlabel(r'major population ($p_1$)')
        ax.set_ylabel(r'$S_\mathrm{rotamer}$')
        ax.set_xlim([1/3,1])
        ax.set_ylim([-.5,1])
        
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
        
        S[S>1]=1
        S[S<1/3]=1/3
        
        p1=np.ones([3,len(S)])*np.nan
        
        #%% First values for S>0.5
        i=S>=0.5
        if i.sum():
            half=len(self.p1)//2
            p1[0][i]=linear_ex(self.S[half:],self.p1[half:],S[i])
            p1[1][i]=linear_ex(self.Seq[half:],self.p1[half:],S[i])
        
        #%% Then values for S<0.5
        i=S<0.5
        if i.sum():
            third=len(self.p1)//3
            p1[1][i]=linear_ex(np.abs(self.Seq[third:]),self.p1[third:],S[i])
            p1[2][i]=linear_ex(np.abs(self.Seq[:third]),(1-self.p1[:third])/2,S[i])
        return p1
        
    
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
        
        #Loop if multiple S/residues provided
        if hasattr(S,'__len__') or hasattr(resid,'__len__'):
            assert hasattr(S,'__len__') and hasattr(resid,'__len__'),"S and resid must have the same length"
            assert len(S)==len(resid),"S and resid must have the same length"
            
            return np.array([self(S0,resid0) for S0,resid0 in zip(S,resid)]).T
        
        which=self.which_pop(resid)
        
        return self.get_pop(S)[which[0 if S>=0.5 else 1]][0],which[0 if S>=0.5 else 1]
        
        
    
    def which_pop(self,resid):
        """
        Determines which assumptions to take on populations based on the results of
        populations from the MD trajectory (HETs_MET_4pw).
        
        Returns two results for each residue in resid. The first result is the
        best model choice if S>=0.5. The second result is the best model 
        choice if S<=0.5

        Parameters
        ----------
        resid : int
            Which resid(s) to investigate.

        Returns
        -------
        int
            0 : assume p1>p2,p3=0
            1 : assume p1>p2=p3
            2 : assume p1=p2>p3
        """
        if not(hasattr(self,'md')):self.md=load_md()[1]
        
        if hasattr(resid,'__len__'):
            return [self.which_pop(res) for res in resid]
        
        pop=self.md[resid]['pop_outer']  #Get the outer population
        
        out=np.zeros(2,dtype=int)
        if pop[1]>0.5*pop[2]:
            out[0]=0   #Assume p1>p2,p3=0
        else:
            out[0]=1   #Assume p1>p2=p3
        if pop[1]/pop[0]<pop[2]/pop[1]: 
            out[1]=1    #Assume p1>p2=p3
        else:
            out[1]=2    #Assume p1=p2>p3
        
        return out        
        # if pop[0]>1/3 and pop[2]>0.5*pop[1]:return 1  #pop[2] a significant fraction of pop[1]
        # if pop[0]>0.5:return 0
        # if pop[1]/pop[0]>pop[2]/pop[1]:return 2  #First two populations equal, large compared to third (third pop is p1)
        # return 1
        
              
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
    md_no,md_corr=load_md()   #These are populations for MD (4pw) without and with methyl correction
    
    #Read in NMR S values, residue numbers and names
    data=pyDR.Project('Projects/directHC')['NMR']['raw'][0]
    S=data.S2**(0.5)*3  #Remove methyl contribution (S*3)
    resids=[int(lbl[:3]) if len(lbl.split(','))==1 else [int(l0[:3]) for l0 in lbl.split(',')] \
            for lbl in data.label]   
    names=[lbl[3:].upper() if len(lbl.split(','))==1 else 'VAL' for lbl in data.label]
    
    #Kick out alanines
    i=np.array(names)!='ALA'
    resids,S,names=np.array(resids,dtype=object)[i],S[i],np.array(names)[i]
     
    #Plot setup
    ax=plt.subplots()[1] 
    w=0.13  #Bar width
    w1=0.25 #For spacing between chi1, chi2
    off=.03
    clrs=(plt.get_cmap('tab10')(7),plt.get_cmap('tab10')(1),plt.get_cmap('tab10')(0))
    

    
    #Get the experimental populations
    res=np.concatenate([np.atleast_1d(res)[:1] for res in resids])
    pop_exp,which=S_v_pop(S, res)
    
    #Now define x-axis labels
    xlabel=list()
    for name,resid,wh0 in zip(names,resids,which):
        if name=='VAL':
            if hasattr(resid,'__len__'):
                xlabel.append(r'$\chi_1$'+'\nV264,\nV267,V275\n')
            else:
                xlabel.append(r'$\chi_1$'+f'\nV{resid}\n\n')
        else:
            xlabel.append(r'$\chi_1$     $\chi_2$'+f'\n{name[0]}{resid}\n\n')
        if wh0==0:xlabel[-1]+=r'$p_3=0$'
        if wh0==1:xlabel[-1]+=r'$p_2=p_3$'
        if wh0==2:xlabel[-1]+=r'$p_1=p_2$'
    
    #%%First, plot valines
    i=names=='VAL'
    
    res=np.concatenate([np.atleast_1d(res)[:1] for res in resids[i]])
    
    
    #Experiment
    ax.bar(np.argwhere(i)[:,0]-w,pop_exp[i],width=w,color=clrs[0],edgecolor='black')
    
    #Simulation (no met corr)
    pop=np.array([np.mean([md_no[res0]['pop_outer'][0] for res0 in res]) if hasattr(res,'__len__') else \
                  md_no[res]['pop_outer'][0] for res in resids[i]])
    ax.bar(np.argwhere(i)[:,0],pop,width=w,color=clrs[1],edgecolor='black')
    
    #Simulation (met corr)
    pop=np.array([np.mean([md_corr[res0]['pop_outer'][0] for res0 in res]) if hasattr(res,'__len__') else \
                  md_corr[res]['pop_outer'][0] for res in resids[i]])
    ax.bar(np.argwhere(i)[:,0]+w,pop,width=w,color=clrs[2],edgecolor='black')
    
    #%% Then plot Leucine/Isoleucine
    i=names!='VAL'
    res=resids[i]
    
    #Experiment
    ax.bar(np.argwhere(i)[:,0]-w-w1,np.ones(i.sum()),width=w,color=clrs[0],edgecolor='black')
    ax.bar(np.argwhere(i)[:,0]-w+w1,pop_exp[i],width=w,color=clrs[0],edgecolor='black')
    
    #Simulation (no met corr)
    pop=np.array([md_no[res0]['pop_inner'][0] for res0 in res])
    ax.bar(np.argwhere(i)[:,0]-w1,pop,width=w,color=clrs[1],edgecolor='black')
    
    pop=np.array([md_no[res0]['pop_outer'][0] for res0 in res])
    ax.bar(np.argwhere(i)[:,0]+w1,pop,width=w,color=clrs[1],edgecolor='black')
    
    #Simulation (met corr)
    pop=np.array([md_corr[res0]['pop_inner'][0] for res0 in res])
    ax.bar(np.argwhere(i)[:,0]-w1+w,pop,width=w,color=clrs[2],edgecolor='black')
    
    pop=np.array([md_corr[res0]['pop_outer'][0] for res0 in res])
    ax.bar(np.argwhere(i)[:,0]+w1+w,pop,width=w,color=clrs[2],edgecolor='black')
    
    ax.set_xticks(range(len(S)))
    ax.set_xticklabels(xlabel)
    ax.set_ylabel(r'major population ($p_1$)')
    ax.figure.set_size_inches([8,5])
    ax.figure.tight_layout()
    
    #%% Plot showing range of values for major population for a given S
    S_v_pop.plot()
    
    #%% Now we make a plot with both possibilities (for SI)
    w=0.25
    ax=plt.subplots()[1]
    pop=S_v_pop.get_pop(S)
    cmap=plt.get_cmap('tab10')
    ax.bar(np.arange(len(S))-w,pop[0],color=cmap(0),width=w,edgecolor='black')
    ax.bar(np.arange(len(S)),pop[1],color=cmap(1),width=w,edgecolor='black')
    ax.bar(np.arange(len(S))+w,pop[2],color=cmap(6),width=w,edgecolor='black')
    ax.legend([r'$p_1>p_2,p_3=0$',r'$p_1>p_2=p_3$',r'$p_1=p_2>p_3$'],loc='lower left')
    
    #Now define x-axis labels
    xlabel=list()
    for name,resid in zip(names,resids):
        if name=='VAL':
            if hasattr(resid,'__len__'):
                xlabel.append(r'$\chi_1$'+'\nV264,\nV267,V275\n')
            else:
                xlabel.append(r'$\chi_1$'+f'\nV{resid}\n\n')
        else:
            xlabel.append(r'$\chi_2$'+f'\n{name[0]}{resid}\n\n')
    ax.set_xticks(range(len(S)))
    ax.set_xticklabels(xlabel)
    ax.figure.set_size_inches([6.4,4.8])
    ax.figure.tight_layout()
    
    #%% Plot results onto molecule
    x=np.zeros(len(data))
    i=np.array(['Ala' not in lbl for lbl in data.label])
    x[i]=1-pop_exp
    data.project.chimera.close()    
    
    data.select.chimera(x=x,norm=False,color=(.1,.1,.1,1))    
    resi=np.concatenate([[str(res0) for res0 in resid] if hasattr(resid,'__len__') else [str(resid)] for resid in resids])
    data.project.chimera.command_line(['~show','show /B:'+','.join(resi)+'&:ILE,LEU,VAL',
                                       '~ribbon','ribbon /B','lighting soft','graphics silhouettes true',
                                       'sel /B&~:ILE,LEU,VAL','show sel','transparency sel 70 atoms','~sel'])
    
    data.project.chimera.savefig('rotamer_pops',options='transparentBackground true')