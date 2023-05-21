#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:33:57 2022

@author: albertsmith
"""


import pyDR
from met_hop_vs_lib import Z2lib
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

linear_ex=pyDR.tools.linear_ex

#%% Load the experimental data
exp=pyDR.Project('directHC')
data=exp['NMR']['proc'][0]

S2hop=1/9
#%% Fit first two detectors to methyl hopping/methyl libration
z2lib=Z2lib()

z,sigma,z0=z2lib.fit2z_sigma(data)  #Hopping log(tc),libration stdev, Hopping log(tc) neglecting libration
S2lib=z2lib.z2S2(z)

zhop=np.linspace(-11.2,-10,101)

# error=list()
# for zhop0 in zhop:    
#     print(zhop0)
#     S2lib=z2lib.z2S2(zhop0)
#     Rmet=linear_ex(data.sens.z,data.sens.rhoz,zhop0)*(S2lib-S2hop)+\
#         data.sens.rhoz[:,0]*(1-S2lib)
#     diff=copy(data)
#     diff.R-=Rmet
#     diff.src_data.S2+=8/9
#     diff=diff.opt2dist()
#     diff.src_data.S2-=8/9
#     diff.R+=Rmet
    
#     error.append(np.abs(diff.R-data.R).sum(axis=1))
  
# i=np.argmin(error,axis=0)

i=np.array([28, 89, 47, 63, 55, 52, 85, 57, 79, 72, 49, 76, 65]) #Result of above optimization (zhop=np.linspace(-11.2,-10,101))
z=zhop[i]


S2=z2lib.z2S2(z)


Rmet=linear_ex(data.sens.z,data.sens.rhoz,z)*(8/9-1+S2)+\
    np.atleast_2d(data.sens.rhoz[:,0]).repeat(len(z),axis=0).T*(1-S2)
diff=copy(data)
diff.R-=Rmet.T
diff.src_data.S2+=8/9   #Scale up the source data S2 (remove methyl influence)
diff=diff.opt2dist()    #Optimize dectector responses (consistent with distribution)
diff.src_data.S2-=8/9   #Put back S2 value to correct amount
diff.R*=9  #Multiply by 9 to counter methyl scaling


out=copy(data)
out.R=Rmet.T          #Contribution from methyl hop/lib
out.R+=diff.R/9     #Contrbution from other motion

exp.append_data(out)

n=9


# sim=pyDR.Project('Frames')['Avg_chi_hop>side_chain_chi']['HETs_MET_4pw']['no_opt'][0]
sim=pyDR.Project('methyl')['Avg_AvOb_methylCC>LF']['no_opt'][0]
target=out.sens.rhoz
sim.detect.r_auto(n,Normalization='I')
# target=np.concatenate((sim.detect.rhoz[:3],target[1:4],sim.detect.rhoz[6:]))
# sim.detect.r_target(target)
sim=sim.fit().opt2dist(rhoz_cleanup=True)

sim1=pyDR.Project('frames_s')['Avg_AvOb_chi_hop>side_chain_chi']['HETs_MET_4pw_SegB']['no_opt'][0]
# sim1.detect.r_target(target)
sim1.detect.r_auto(n)
sim1=sim1.fit().opt2dist(rhoz_cleanup=True)

# sim=pyDR.Project('proj_md')['OuterHop'][1]

from Fig4_plot_chi_pop import load_md
_,md_corr=load_md()

def plot_residue(resid,ax=None):
    if ax is None:ax=plt.figure().add_subplot(111)
    
    
    i=[str(resid)==lbl[:3] for lbl in diff.label]
    
    label=diff.label[i][0]
    
    #Plot simulation
    if label[3:]!='Ala':
        ii=-2
        # hdl=sim.sens.plot_rhoz(index=range(sim.sens.rhoz.shape[0]+ii),color='lightgrey',ax=ax)
        z=[-11.45,*sim.sens.info['z0'][1:]]     
        if label[3:]=='Ile':
            index=[lbl[:3]==label[:3] and lbl[4:6]=='CD' for lbl in sim.label]
        elif ',' in label:
            index=[lbl[:3] in [lbl0[:3] for lbl0 in label.split(',')] for lbl in sim.label]
        else:
            index=[lbl[:3]==label[:3] for lbl in sim.label]
        R=sim.R[index].mean(axis=0)
        R1=sim1.R[index].mean(axis=0)
        w=.15
        
        
        coords=np.concatenate(([[*z[:ii],z[ii-1],z[0]]],[[*R[:ii],0,0]]),axis=0).T
        polygons = [];
        polygons.append(Polygon(coords))
        p = PatchCollection(polygons, alpha=0.7)
        p.set_facecolor('lightgrey')
        ax.add_collection(p)
        coords=np.concatenate(([[*z[:ii],z[ii-1],z[0]]],[[*R1[:ii],0,0]]),axis=0).T
        polygons = [];
        polygons.append(Polygon(coords))
        p = PatchCollection(polygons, alpha=0.7)
        p.set_facecolor('darkgrey')
        ax.add_collection(p)
        # for z0,R0 in zip(z,R):ax.bar(z0,R0,width=w,facecolor='black',edgecolor='grey')
        # for z0,R0 in zip(z,R1):ax.bar(z0,R0,width=w,facecolor='grey',edgecolor='grey')
        
        
        
        # pop=md_corr[int(label[:3])]['pop_outer']
        # S2=(pop**2).sum()-2/3*(pop[0]*pop[1]+pop[0]*pop[2]+pop[1]*pop[2])
        # ax.plot(ax.get_xlim(),(1-S2)*np.ones(2),color='black',linestyle='--')
        
        ax.set_xlim([-12,z[ii-1]])
    else:
        ax.set_xlim([-12,-7])
    
    #Plot experiment
    z=[-10.8,*diff.sens.info['z0'][1:4],-7.9]
    R=diff.R[i,[0,1,2,3,0,4,5]]
    clr=plt.get_cmap('tab10')([0,1,2,3,0,4,5])
    if R[1]>R[3]:clr[4]=plt.get_cmap('tab10')(7)
    else:clr[0]=plt.get_cmap('tab10')(7)
    clr[:,-1]=.75
    w=.3
    for z0,R0,clr0 in zip(z,R,clr):ax.bar(z0,R0,width=w,facecolor=clr0,edgecolor='black')

    
    
    ax.plot(ax.get_xlim(),(1-diff.src_data.S2[i]-8/9)*np.ones(2)*9,color='black',linestyle=':')
    
    
    ax.set_ylim([0,.9])
    
    ax.set_xticks(range(-12,-6))
    ax.set_xticklabels([None if not(z%2) else ('{:.0f} ps'.format(10**z*1e12) if z<-9 else '{:.0f} ns'.format(10**z*1e9)) for z in range(-12,-6)]) 
    ax.set_xlabel(r'$\tau_c$')
    ax.text(-11.5,0.8,label)
    

resids=list()
for lbl in diff.label:
    if lbl[3:]!='Ala':resids.append(int(lbl[:3]))

# resids=[int(lbl[:3]) for lbl in diff.label]

fig=plt.figure()
ax=[fig.add_subplot(3,3,k+1) for k in range(9)] if len(resids)<=9 else [fig.add_subplot(4,4,k+1) for k in range(13)]

for r,a in zip(resids,ax):plot_residue(r,ax=a)
fig.set_size_inches([12.5,9])
fig.tight_layout()


exp.chimera.saved_commands=['~show','show /B:'+','.join([str(res) for res in resids]),'ribbon /B',
              'turn y -15','turn z 20','turn y -90','turn x -35','turn z 5',
              'view','zoom 1.4','lighting soft','graphics silhouettes true']
for k in range(1,4):
    exp.chimera.close()
    diff.chimera(rho_index=k,scaling=1.5)
    exp.chimera.savefig('/Users/albertsmith/Documents/Dynamics/HETs_Methyl_Loquet'+\
                        f'/Paper/Figures/figure_parts/exp_nomet_rho{k}.png',
                        options='transparentBackground true')

    
    

