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
from pyDR.Frames import Chunk

pyDR.Defaults['reduced_mem']=True
"""
The above global option causes Frame Objects, data objects, and Projects to
discard large, partially processed arrays in order to save memory. This will
remove our ability to re-process some of this data
(recommended for final processing of large MD trajectories, but not recommended
 for testing out processing)
"""


"""
The 10 microsecond trajectories sampled every 5 ps create some challenges for
data analysis, mainly that there are 2000000 time points. We do not really need
to process all points simultaneously, so we have two projects here. The first,
frames, takes the trajectory in five chunks, each sampled every 5 ps for 2 
microseconds. This is used for analyzing faster, sub-microsecond motion. The 
second, proj_s, also takes the trajectory in five chunks, only takes every
fifth time point for the full 10 microseconds. This is repeated five times, 
starting with time point 0,1,2,3, and 4. This project is used for analyzing
slow motion, where only taking every fifth point causes us to lose information
about fast motion.

Note that this chunking is only done for HETs_MET_4pw_SegB, for which we have
10 microseconds of simulation, but not for HETs_4pw, which only contains 2 
microseconds of simulation
"""
proj=pyDR.Project('Project/frames',create=True)
proj_s=pyDR.Project('Project/frames_s',create=True)


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
    
    sel=dict(Nuc='ivla',segids='B')
    
    frames=list()
    frames.append({'Type':'hops_3site','sigma':.005,**sel})
    frames.append({'Type':'methylCC','sigma':.005,**sel})
    frames.append({'Type':'chi_hop','n_bonds':1,'sigma':.05,**sel})
    frames.append({'Type':'side_chain_chi','n_bonds':1,'sigma':.05,**sel})
    frames.append({'Type':'chi_hop','n_bonds':2,'sigma':.05,**sel})
    frames.append({'Type':'side_chain_chi','n_bonds':2,'sigma':.05,**sel})
    
    for topo,traj in zip(topos,trajs):
        
        select=pyDR.MolSelect(topo=topo,traj_files=traj,project=proj)
        select.select_bond(**sel) 
            
        fr_obj=pyDR.Frames.FrameObj(select)
        fr_obj.tensor_frame(sel1=1,sel2=2)
        
        for f in frames:fr_obj.new_frame(**f)

        if topo==topo[1]:        

            fr_obj.project=proj_s
            with Chunk(fr_obj,mode='slow',chunks=5) as chunk:
                chunk()
            proj_s.save()
            fr_obj.project=proj
            with Chunk(fr_obj,mode='fast',chunks=5) as chunk:
                chunk()
            proj.save()
        else:
            fr_obj.load_frames()
            fr_obj.post_process()
            fr_obj.frames2data()
            

            proj[-9:].detect.r_no_opt(20)  
            proj[-9:].fit()
            proj.save()
        
    
    
#%% Processing    
if len(proj)<36: 
    "First we average together equivalent bonds (3 bonds in each methyl group)"
    avgMethyl(proj)
    avgMethyl(proj_s)
    proj.save()
    proj_s.save()

if len(proj)<54:    
    "Here we process using detectors optimized for describing all information in MD"
    proj['.+Avg'].detect.r_auto(8) #Optimize 8 detectors (these will be different between the two trajectories)
    proj['.+Avg'].fit(bounds=True)
    proj.save()
    
    proj_s['.+Avg'].detect.r_auto(8) #Optimize 8 detectors (these will be different between the two trajectories)
    proj_s['.+Avg'].fit(bounds=True)
    proj_s.save()

if len(proj)<72:
    proj['proc'].opt2dist(rhoz_cleanup=True) #This improves the fit and cleans negative parts of the sensitivites
    proj.save()
    proj_s['proc'].opt2dist(rhoz_cleanup=True) #This improves the fit and cleans negative parts of the sensitivites
    proj_s.save()

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
In the above analysis, we create two relatively large projects (72-144 data sets). For
practical use, we would like to have something a little easier to deal with,
and also with more obvious titles. Also, note that based on the frame definitions,
chi1/chi2 motions are sorted into outer/inner bond rotations as opposed to 
chi1/chi2 bond rotations. These will be resorted as well.

Note, we also have one data object called OuterHop, which contains just hopping
motion of the outermost bond, whether that is the chi1 or chi2 bond
"""

proj0=[proj,proj,proj,proj_s]
proj_titles=['proj_md','proj_exp','proj_exp_r0e','proj_md_s']
sub_strs=['o8.','o12.','o13.','o8.']
titles=['Direct','Product','MetLib','MetHop','Chi2Lib','Chi2Hop','Chi1Lib','Chi1Hop','CaCb','OuterHop']
resnames=[s[0].resname for s in proj[0].select.sel1[::3]]
names=[s[0].name for s in proj[0].select.sel1[::3]]
chi2_ind=np.array([(resname in ['ILE','LEU'] and 'CD' in name) for resname,name in zip(resnames,names)],dtype=bool)
chi1_ind=np.array(['CG' in name for name in names],dtype=bool)
outer_ind=np.array([(resname in ['ILE','LEU'] and 'CD' in name) or (resname in 'VAL')\
                    for resname,name in zip(resnames,names)],dtype=bool)

for sub_str,proj_title,Proj in zip(sub_strs,proj_titles,proj0):
    proj2=pyDR.Project(os.path.join('Projects',proj_title),create=True)
    if len(proj2)<20:
        for traj in trajs:
            sub=Proj[sub_str][os.path.split(traj)[1].split('.')[0]]  #18 data sets (either optimized for MD or for comparison to experiment)
            if len(sub):
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
    topo=topos[1]
    traj=trajs[1]
    
    
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
    
    with Chunk(fr_obj,mode='slow',chunks=5,n=15) as chunk:
        chunk()
    
    # fr_obj.load_frames()
    # fr_obj.post_process()
    # fr_obj.frames2data()
    
    # key=os.path.split(traj)[1].split('.')[0]
    # proj[key].detect.r_no_opt(20)
    # proj[key].fit(bounds=False)
    proj.save()
    
    
    avgMethyl(proj)
    proj.save()
    
    
    