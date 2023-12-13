#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:20:13 2023

@author: albertsmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:14:42 2022

@author: albertsmith
"""

import pyDR
import os
import numpy as np
import matplotlib.pyplot as plt

vft=pyDR.MDtools.vft

md_dir='/Volumes/My Book/HETs/'

topos=[os.path.join(md_dir,topo) for topo in ['HETs_3chain.pdb','HETs_SegB.pdb']]
trajs=[os.path.join(md_dir,os.path.join('MDSimulation',traj)) for traj in ['HETs_4pw.xtc','HETs_MET_4pw_SegB.xtc']]


#%% Load vectors and calculate S for both trajectories

filename='chi12_populations/S_{}.npy'

for topo,traj in zip(topos,trajs):
    if os.path.exists(filename.format(os.path.split(traj)[1].split('_',maxsplit=1)[1][:-4])):continue
    select=pyDR.MolSelect(topo=topo,traj_files=traj)
    select.select_bond(Nuc='ivlar',segids='B')
    
    select.traj.step=10
    select.traj.ProgressBar=True
    
    v=np.array([vft.norm(select.v.T).T for _ in select.traj]).T
    
    D2=vft.D2(*vft.getFrame(v)).reshape([5,len(select)//3,3,len(select.traj)]).mean(2).mean(-1)
    S=vft.Spher2pars(D2*np.sqrt(3/2))[0]
    
    np.save(filename.format(os.path.split(traj)[1].split('_',maxsplit=1)[1][:-4]),S,allow_pickle=False)
    label=[int(lbl.split('_')[0]) for lbl in select.label[::3]]
    np.save('chi12_populations/S_lbl.npy',label,allow_pickle=False)
    
    
#%% Plot order parameters



data=pyDR.Project('Projects/directHC')['NMR']['raw']
label=[[int(lbl[:3]) for lbl in label.split(',')] for label in data.label]
Sexp=np.sqrt(data.S2)

sim_lbl=np.load('chi12_populations/S_lbl.npy',allow_pickle=False)
i=np.array([np.isin(sim_lbl,lbl) for lbl in label])
i=(i.T/i.sum(1)).T

S_4pw=i @ np.load('chi12_populations/S_4pw.npy',allow_pickle=False)

S_MET_4pw=i @ np.load('chi12_populations/S_MET_4pw_SegB.npy',allow_pickle=False)


j=np.array(['Ala' not in lbl for lbl in data.label])

w=0.25
fig,ax=plt.subplots()
cmap=plt.get_cmap('tab10')
ax.bar(np.arange(j.sum())-w,np.abs(Sexp[j]),width=w,color=cmap(7))
ax.bar(np.arange(j.sum()),np.abs(S_4pw[j]),width=w,color=cmap(1))
ax.bar(np.arange(j.sum())+w,np.abs(S_MET_4pw[j]),width=w,color=cmap(0))
ax.set_xticks(np.arange(j.sum()))
label=np.array(data.label[j])
label[5]=label[5][:13]+'\n'+label[5][14:]
ax.set_xticklabels(label,rotation=90)
ax.set_ylabel('S')
fig.tight_layout()