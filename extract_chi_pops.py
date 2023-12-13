#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:14:42 2022

@author: albertsmith
"""

import pyDR
import os
import numpy as np

vft=pyDR.MDtools.vft

topos=['HETs_3chain.pdb','HETs_SegB.pdb']
trajs=['HETs_4pw.xtc','HETs_MET_4pw_SegB.xtc']

#%% Initial frame analysis
"""
Note that we set up this script so that if data is not already stored in the
project, then the full processing will run. However, we have provided a project
file that contains all data sets already processed. Then the various if statements
(if len(proj)<...) will prevent any of the code from actually running.
"""    


md_dir='/Volumes/My Book/HETs/MDSimulation'
topo='HETs_3chain.pdb'

topos=[os.path.join(os.path.split(md_dir)[0],topo) for topo in topos]
trajs=[os.path.join(md_dir,traj) for traj in trajs]


#Define the frames used to extract angles
sel=dict(Nuc='ivla',segids='B')

frames=list()
# frames.append({'Type':'hops_3site','sigma':.005,**sel})
# frames.append({'Type':'methylCC','sigma':.005,**sel})
frames.append({'Type':'chi_hop','n_bonds':1,'sigma':.05,**sel})
frames.append({'Type':'side_chain_chi','n_bonds':1,'sigma':.05,**sel})
frames.append({'Type':'chi_hop','n_bonds':2,'sigma':.05,**sel})
frames.append({'Type':'side_chain_chi','n_bonds':2,'sigma':.05,**sel})

for topo,traj in zip(topos,trajs):
    select=pyDR.MolSelect(topo=topo,traj_files=traj)
    select.select_bond(**sel)
    
    fr_obj=pyDR.Frames.FrameObj(select)
    fr_obj.tensor_frame(sel1=1,sel2=2)
    
    for f in frames:fr_obj.new_frame(**f)
    fr_obj.load_frames()
    fr_obj.post_process()
    
    vecs=fr_obj.vecs['v']   #Vectors giving the frame orientations
    
    vr=fr_obj.frame_info['info'][0]['vr']   #Reference vectors for the outer chi angle
    "Each vector in v for a given residue should match one of the three reference vectors for that residus"
    v=vft.applyFrame(vecs[0][0],nuZ_F=vecs[1][0],nuXZ_F=vecs[1][1])  #Aligned vectors for chi hopping
    
    """We sweep over the three reference vectors (vr0 in vr) and determine which
    of the three vectors has the largest overlap with v for every time point
    (one should be in full overlap, i.e. the sum should yield 1)
    """
    state=np.argmax([(vr0.T*v.T).sum(axis=-1) for vr0 in vr],axis=0)  
    
    "The above skips alanines. Here we fill in alanine data with NaN"
    pop_inner0=np.array([(state==k).sum(0)/state.shape[0] for k in range(3)]).T
    i=np.logical_not(np.isnan(fr_obj.frame_info['frame_index'][0][::3]))
    pop_inner=np.ones([i.shape[0],3])*np.nan
    pop_inner[i]=pop_inner0
        
    
    "Same procedure as above for outer angle"
    vr=fr_obj.frame_info['info'][2]['vr']   #Reference vectors for the outer chi angle
    v=vft.applyFrame(vecs[2][0],nuZ_F=vecs[3][0],nuXZ_F=vecs[3][1])  #Aligned vectors for chi hopping
    state=np.argmax([(vr0.T*v.T).sum(axis=-1) for vr0 in vr],axis=0)  
    
    "Skips alanines/valines"
    pop_outer0=np.array([(state==k).sum(0)/state.shape[0] for k in range(3)]).T
    i=np.logical_not(np.isnan(fr_obj.frame_info['frame_index'][2][::3]))
    pop_outer=np.ones([i.shape[0],3])*np.nan
    pop_outer[i]=pop_outer0
    
    
    "Write out the results"
    resids=fr_obj.select.sel1[::3].resids
    resnames=fr_obj.select.sel1[::3].resnames
    
    if not(os.path.exists('chi12_populations')):os.mkdir('chi12_populations')
    
    filename='chi12_populations/chi12_'+traj.split('/')[-1].split('.')[0]+'.txt'
    
    with open(filename,'w') as f:
        f.write('Resid Resname PopOuter [,,] PopInner [,,]\n')
        for resid,resname,pi,po in zip(resids,resnames,pop_inner,pop_outer):
            f.write(f'{resid} {resname} [')
            f.write(','.join(['-1' if np.isnan(p) else f'{p:.3f}' for p in pi])+'], [')
            f.write(','.join(['-1' if np.isnan(p) else f'{p:.3f}' for p in po])+']\n')
        