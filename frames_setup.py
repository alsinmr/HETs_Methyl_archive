#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:55:17 2022

@author: albertsmith
"""

import pyDR
import os
import numpy as np
from pyDR.misc.Averaging import avgMethyl
from copy import copy
from pyDR.misc.Averaging import appendDataObjs



proj=pyDR.Project('frames',create=True)


md_dir='/Volumes/My Book/HETs/'
topos=['HETs_3chain.pdb','HETs_SegB.pdb']
trajs=['MDSimulation/HETs_4pw.xtc','MDSimulation/HETs_MET_4pw_SegB.xtc']

topos=[os.path.join(md_dir,topo) for topo in topos]
trajs=[os.path.join(md_dir,traj) for traj in trajs]

#%% Initial frame analysis
"""
Note that we set up this script so that if data is not already stored in the
project, then the full processing will run. However, we have provided a project
file that contains all data sets already processed. Then the various if statements
(if len(proj)<...) will prevent any of the code from actually running.
"""    

if len(proj)<18:  #Skip this if already run
    
    
    resids=np.array([219, 221, 222, 223, 228, 231, 237, 239, 241, 244, 245, 247, 248,
           249, 250, 254, 256, 264, 267, 268, 275, 276, 277])
    
    sel=dict(Nuc='ivla',segids='B')
    
    frames=list()
    frames.append({'Type':'hops_3site','sigma':.005,**sel})
    frames.append({'Type':'methylCC','sigma':.005,**sel})
    frames.append({'Type':'chi_hop','n_bonds':1,'sigma':.05,**sel})
    frames.append({'Type':'side_chain_chi','n_bonds':1,'sigma':.05,**sel})
    frames.append({'Type':'chi_hop','n_bonds':2,'sigma':.05,**sel})
    frames.append({'Type':'side_chain_chi','n_bonds':2,'sigma':.05,**sel})
    
    for topo,traj in zip(topos,trajs):
        
        proj0=pyDR.Project('temp',create=True)
        select=pyDR.MolSelect(topo=topo,traj_files=traj,project=proj0,step=100)
        for k in range(4): #Process in 4 chunks (memory usage is pretty high for 10 microsecond trajectory)
            resi0=resids[k*6:(k+1)*6]
            select.select_bond(**sel,resids=resi0) 
            
            fr_obj=pyDR.Frames.FrameObj(select)
            fr_obj.tensor_frame(sel1=1,sel2=2)
            
            for f in frames:fr_obj.new_frame(**f,resids=resi0)
            fr_obj.load_frames()
            fr_obj.post_process()
            fr_obj.frames2data()
            

            proj0[-9:].detect.r_no_opt(20)  #Change back to 20!
            proj0[-9:].fit()
            proj0.clear_memory()
            
        
        for title in np.unique(proj0.titles):
            data=[*proj0[title]]
            for d in data: #Disconnect data from the project, also get rid of the source data
                d.project=None
            data=appendDataObjs(data)
            data.source.additional_info=data.source.additional_info[5:]  #Just shortens the title a little bit            
            proj.append_data(data)
        proj.save()
        
    
    
#%% Processing    
if len(proj)<36: 
    "First we average together equivalent bonds (3 bonds in each methyl group)"
    avgMethyl(proj)
    proj.save()

if len(proj)<54:    
    "Here we process using detectors optimized for describing all information in MD"
    proj['.+Avg'].detect.r_auto(8) #Optimize 8 detectors (these will be different between the two trajectories)
    proj['.+Avg'].fit(bounds=True)
    proj.save()

if len(proj)<72:
    proj['proc'].opt2dist(rhoz_cleanup=True) #This improves the fit and cleans negative parts of the sensitivites
    proj.save()

if len(proj)<108:
    "Here we process to match the experimental detector sensitivities"
    nmr=pyDR.IO.readNMR('HETs_13C.txt').sens
    r=pyDR.Sens.Detector(nmr)
    r.r_auto(5).inclS2()
    target=r.rhoz[:4]
    target[0,94:]=0
    proj['no_opt']['.+Avg'].detect.r_target(target,n=12)
    proj['no_opt']['.+Avg'].fit(bounds=True)
    proj['p12.+Avg'].opt2dist()  #This improves the fit (but we do not want to clean the detectors since they should match experiment)
    proj.save()
    
if len(proj)<144:
    "Here we process to match experiment AND include slower (>~1 ns) motions in rho0"
    nmr=pyDR.IO.readNMR('HETs_13C.txt').sens
    r=pyDR.Sens.Detector(nmr)
    r.r_auto(5).inclS2()
    target=r.rhoz[:4]
    target[0,130:145]=target[0,130]*np.linspace(1,0,15)
    target[0,145:]=0
    proj['no_opt']['.+Avg'].detect.r_target(target,n=13)
    proj['no_opt']['.+Avg'].fit(bounds=True)
    proj['p13.+Avg'].opt2dist()
    proj.save()




#%% Re-sorting / new save
"""
In the above analysis, we create a relatively large project (108 data sets). For
practical use, we would like to have something a little easier to deal with,
and also with more obvious titles. Also, note that based on the frame definitions,
chi1/chi2 motions are sorted into outer/inner bond rotations as opposed to 
chi1/chi2 bond rotations. These will be resorted as well.

Note, we also have one data object called OuterHop, which contains just hopping
motion of the outermost bond, whether that is the chi1 or chi2 bond
"""


proj_titles=['proj_md','proj_exp','proj_exp_r0e']
sub_strs=['o8.','o12.','o13.']
titles=['Direct','Product','MetLib','MetHop','Chi2Lib','Chi2Hop','Chi1Lib','Chi1Hop','CaCb','OuterHop']
resnames=[s[0].resname for s in proj[0].select.sel1[::3]]
names=[s[0].name for s in proj[0].select.sel1[::3]]
chi2_ind=np.array([(resname in ['ILE','LEU'] and 'CD' in name) for resname,name in zip(resnames,names)],dtype=bool)
chi1_ind=np.array(['CG' in name for name in names],dtype=bool)
outer_ind=np.array([(resname in ['ILE','LEU'] and 'CD' in name) or (resname in 'VAL')\
                    for resname,name in zip(resnames,names)],dtype=bool)

for sub_str,proj_title in zip(sub_strs,proj_titles):
    proj2=pyDR.Project(proj_title,create=True)
    if len(proj2)<20:
        for traj in trajs:
            sub=proj[sub_str][traj]  #18 data sets (either optimized for MD or for comparison to experiment)
            for k,title in enumerate(titles):
                if 'Chi2' in title:
                    data=copy(sub[k])
                    data.del_data_pt(np.argwhere(np.logical_not(chi2_ind))[:,0])
                elif 'Chi1' in title:
                    data=copy(sub[k-2])
                    data.R[chi2_ind]=sub[k].R[chi2_ind]
                    data.Rstd[chi2_ind]=sub[k].Rstd[chi2_ind]
                    data.del_data_pt(np.argwhere(np.logical_not(np.logical_or(chi1_ind,chi2_ind))))
                elif 'OuterHop'==title:
                    data=copy(sub[5])
                    data.del_data_pt(np.argwhere(np.logical_not(outer_ind))[:,0])
                else:
                    data=copy(sub[k])
                if proj_title=='proj_exp':
                    data.del_exp(range(4,12))
                data.label=np.array([lbl.rsplit('_',maxsplit=1)[0] for lbl in data.label],dtype='<U12')
                data.src_data=None
                data.source.additional_info=title
                data.source.saved_filename=None
                proj2.append_data(data)
        proj2.save()
  
#%% One more frame analysis, where we only factor out methyl motion    


proj=pyDR.Project('methyl',create=True)
      
if not(os.path.exists('methyl')):  #Skip this if already run
    md_dir='/Volumes/My Book/HETs/MDSimulation'
    topo='HETs_3chain.pdb'
    
    trajs=[os.path.join(md_dir,traj) for traj in trajs]
    
    
    sel=dict(Nuc='ivla',segids='B')
    
    frames=list()
    frames.append({'Type':'hops_3site','sigma':.005,**sel})
    frames.append({'Type':'methylCC','sigma':.005,**sel})

    traj=trajs[1]

    select=pyDR.MolSelect(topo=topo,traj_files=traj,project=proj)
    select.select_bond(**sel)
    
    fr_obj=pyDR.Frames.FrameObj(select)
    fr_obj.tensor_frame(sel1=1,sel2=2)
    
    for f in frames:fr_obj.new_frame(**f)
    fr_obj.load_frames()
    fr_obj.post_process()
    fr_obj.frames2data()
    
    key=os.path.split(traj)[1].split('.')[0]
    proj[key].detect.r_no_opt(20)
    proj[key].fit(bounds=False)
    proj.save()
    
    
    avgMethyl(proj)
    
    nmr=pyDR.IO.readNMR('HETs_13C.txt').sens
    r=pyDR.Sens.Detector(nmr)
    r.r_auto(5).inclS2()
    target=r.rhoz[:4]
    target[0,130:145]=target[0,130]*np.linspace(1,0,15)
    target[0,145:]=0
    
    proj['.+Avg'].detect.r_target(target,n=12)
    proj['.+Avg']['no_opt'].fit(bounds=True)
    proj['.+Avg']['proc'].opt2dist(rhoz_cleanup=True)
    
    