#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:51:43 2022

@author: albertsmith
"""

"""
In this script, we test via MD simulation whether a number of models are 
reasonable for fitting motions in the  experimental data. Analysis will be
performed on frame analysis results for methyl motion
"""

import pyDR
import numpy as np
from pyDR.Fitting.fit import model_free

proj=pyDR.Project('proj_md')  #Just analyze the methyl corrected simulation

"We will only check model validity for residues that we have experimental data"
resids=np.concatenate([[int(lbl0[:3]) for lbl0 in lbl.split(',')] if ',' in lbl else \
        [int(lbl[:3])] for lbl in pyDR.Project('directHC')['NMR']['raw'].label])

index=list()
for lbl in proj[0].label:
    if int(lbl[:3]) in resids:  #Experimental data for this residue
        if lbl[4:] in ['CB','CG1','CD','CD1','CD2']:     #Ala CB, Val CG1, Ile CD, Leu CD1/CD2
            index.append(True)
        elif lbl[4:]=='CG2' and lbl[:3]+'_CG1' in proj[0].label: #Val CG2 (need to exclude Ile CG2)
            index.append(True)
        else:
            index.append(False)
    else:
        index.append(False)
index=np.array(index)
            

#%% One parameter fit for methyl rotation
"""
Here we test if the methyl rotation dynamics can be reasonably well fit based
on a single correlation time and a fixed amplitude (S2=1/9, i.e. A=8/9)
"""

data=proj['HETs_MET_4pw']['MetHop'][0]
z,*_,fit=model_free(data,nz=1,fixA=[8/9])

data.plot(style='bar',index=index)
fit.plot()

for a in proj.plot_obj.ax[:3]:a.set_ylim([0,.95])
for a in proj.plot_obj.ax[3:-1]:a.set_ylim([0,.1])
proj.plot_obj.show_tc()

proj.fig.set_size_inches([5.8,8.5])

"""
Second we test if the methyl rotation dynamics can be reasonably well fit based
on a single correlation time and a variable amplitude (S2=1/9, i.e. A=8/9)
"""

z,A,_,fit=model_free(data,nz=1)

proj.current_plot=2
data.plot(style='bar',index=index)
fit.plot()

for a in proj.plot_obj.ax[:3]:a.set_ylim([0,.95])
for a in proj.plot_obj.ax[3:-1]:a.set_ylim([0,.1])
proj.plot_obj.show_tc()

proj.fig.set_size_inches([5.8,8.5])

"""
Some small amplitude mis-fitting occurs in rho3 (1.3 ns), probably attributed
to occasional changes in the correlation time due to configurational changes. 
Fitting is not improved by allowing a variable amplitude, so that we believe
the one parameter (correlation time), is the best model for this motion
"""

#%% One parameter fit for methyl libration
"""
Here we test if the methyl libration dynamics can be reasonably well fit based
on a single amplitude with arbitrarily fast correlation time (set z=log(tc)=-14)
"""

data=proj['HETs_MET_4pw']['MetLib'][0]
_,A,_,fit=model_free(data,nz=1,fixz=[-14])

proj.current_plot=3
data.plot(style='bar',index=index)
fit.plot()

proj.plot_obj.ax[0].set_ylim([0,.5])
for a in proj.plot_obj.ax[1:-1]:a.set_ylim([0,.05])
proj.plot_obj.show_tc()

proj.fig.set_size_inches([5.8,8.5])

"""
The majority of methyl libration responses are well-fit by an arbitrarily fast 
motion described only by an amplitude. However, the total amplitude of motion is
found predominantly in rho0. Small responses for other detectors are present,
and we attribute these to coupling of the librational dynamics to other slower 
motions. These small responses are not fit by this model.
"""

#%% One parameter fit for Outer hopping motion
data=proj['HETs_MET_4pw']['OuterHop'][0]

from Fig3C_plot_chi_pop import load_md
_,md_corr=load_md()

index=list()  #We need a new index since no Alanines in this data set
for lbl in data.label:
    if int(lbl[:3]) in resids:  #Experimental data for this residue
        if lbl[4:] in ['CB','CG1','CD','CD1','CD2']:     #Ala CB, Val CG1, Ile CD, Leu CD1/CD2
            index.append(True)
        elif lbl[4:]=='CG2' and lbl[:3]+'_CG1' in proj[0].label: #Val CG2 (need to exclude Ile CG2)
            index.append(True)
        else:
            index.append(False)
    else:
        index.append(False)
index=np.array(index)

"We'll use the populations of the outer rotamer to fix the total amplitude"
pop=np.array([md_corr[int(lbl[:3])]['pop_outer'] for lbl in data.label]).T
S2=(pop**2).sum(0)-2/3*(pop[0]*pop[1]+pop[0]*pop[2]+pop[1]*pop[2])
    
shift=S2-data.R[:,-1]
shift[shift<0]
i=shift>data.R[:,-2]
shift[i]=data.R[i,-2]
data.R[:,-1]+=shift
data.R[:,-2]-=shift

z,A,_,fit=model_free(data,nz=1,fixA=[1-S2])

proj.current_plot=4
data.plot(style='bar',index=index)
fit.plot()

for a in proj.plot_obj.ax[:3]:a.set_ylim([0,.95])
for a in proj.plot_obj.ax[3:-1]:a.set_ylim([0,1])
proj.plot_obj.show_tc()

proj.fig.set_size_inches([5.8,8.5])

"""
Not a particularly impressive fit. Clearly, this model misses some aspect of the
motion
"""

#%% Now try three parameter fit (extract exchange matrix)

from fit_biexp_2ex import Biexp2ex

biexp2ex=Biexp2ex()

k,fit=biexp2ex.fit(pop,data)

proj.current_plot=5
data.plot(style='bar',index=index)
fit.plot()

for a in proj.plot_obj.ax[:3]:a.set_ylim([0,.95])
for a in proj.plot_obj.ax[3:-1]:a.set_ylim([0,1])
proj.plot_obj.show_tc()

proj.fig.set_size_inches([5.8,8.5])

"""
Does not lead to remarkable improvement compared to the previous plot, despite
introducing two more variables. Resulting matrices are also not so clearly
stable.
"""