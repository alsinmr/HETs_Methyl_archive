#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:48:26 2022

@author: albertsmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:55:17 2022

@author: albertsmith
"""

import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')
import pyDR
import os



#%% Pre-processing
md_dir='/Volumes/My Book/HETs/MDSimulation'
topos=[*['HETs_3chain.pdb' for _ in range(5)],'HETs_2kj3.pdb','HETs_5chain_B.pdb']
trajs=[os.path.join(md_dir,traj) for traj in \
       ['HETs_3pw.xtc','HETs_MET_3pw.xtc','HETs_MET_tip3p_870ns.xtc','HETs_4pw.xtc','HETs_MET_4pw.xtc',
        'HETs_MET_4pw_2kj3.xtc','HETs_MET_4pw_5chain.xtc']]

projHC=pyDR.Project('Projects/directHC',create=True)
projHN=pyDR.Project('Projects/directHN',create=True)
    
for proj in [projHC,projHN]:  #Loop over the two projects
    sub=proj['no_opt']-proj['Avg']
    for topo,traj in zip(topos,trajs):
        if traj in [d.select.traj.files[0] for d in sub]:continue #Check if this has already been run
        print(traj)
        proj.clear_memory() #Reduce memory usage
        select=pyDR.MolSelect(topo=topo,traj_files=traj,project=proj)
        select.select_bond(Nuc='ivla' if proj is projHC else '15N',segids='B')
        
        pyDR.md2data(select)
        proj[-1].detect.r_no_opt(20)
        proj[-1].fit(bounds=False)
    proj.save()

    


#%% Load NMR data and add selection definition for 13C
if len(projHC['raw']['NMR'])==0:  #Check to see if raw nmr data already loaded
    projHC.append_data('HETs_13C.txt')  #Load the methyl relaxation data
    projHC[-1].del_data_pt(range(-4,0))   #These are unassigned peaks that we delete
    
    "Create the selection for the experimental data (still pretty messy)"
    select=pyDR.MolSelect(topo='HETs_3chain.pdb') #Load a structure
    select._mdmode=True #mdmode: sometimes convenient for dealing with atom selections
    
    resids=list()
    for lbl in projHC['NMR'][0].label:
        resids.append([int(l[:3]) for l in lbl.split(',')])
        
    sel1=list()
    sel2=list()
    for r in resids:
        for k,r0 in enumerate(r):
            select.select_bond(Nuc='ivla',resids=r0,segids='B')
            if select.sel1.residues.resnames[0]=='ILE':select.select_bond(Nuc='ivlal',resids=r0,segids='B')
            if k==0:
                sel1.append(select.sel1)
                sel2.append(select.sel2)
            else:
                sel1[-1]+=select.sel1
                sel2[-1]+=select.sel2
    
    select._mdmode=False            
    select.sel1=sel1
    select.sel2=sel2
    
    projHC['NMR'].select=select
    projHC.save()

#%% Load NMR data and add selection definition for 15N
if len(projHN['raw']['NMR'])==0:
    projHN.append_data('HETs_15N.txt')
    select=pyDR.MolSelect(topo='HETs_3chain.pdb')
    select.select_bond(Nuc='15N',segids='B',resids=projHN[-1].label)
    projHN['NMR'].select=select
    projHN.save()

#%% Fitting and comparison
for proj in [projHC,projHN]:
    if len(proj['NMR']['proc'])==0:
        proj['NMR']['raw'].detect.r_auto(5)
        proj['NMR']['raw'].detect.inclS2()
        proj['NMR']['raw'].fit()
        
    "Target functions to match experimental NMR sensitivities"
    if proj is projHC:
        target=proj['NMR']['proc'][-1].sens.rhoz[:4]
        target[0,94:]=0
    else:
        target=proj['NMR']['proc'][-1].sens.rhoz[:3]
        target[0,94:]=0
    
    
    "We average together MD data to match experimental selections"
    from pyDR.misc.Averaging import avg2sel
    sub=proj['no_opt']['Avg']
    for d in proj['no_opt']-sub:
        if d.source.short_file not in sub.short_files:
            avg2sel(d,proj['NMR']['raw'].select)
    
    
    sub=proj['MD']['Avg']['proc']
    proj['MD']['Avg'].detect.r_target(target=target,n=12)
    for d in proj['MD']['Avg']-sub-proj['opt_fit']:
        if d.source.short_file not in sub.short_files:
            d.fit(bounds=False).opt2dist(rhoz_cleanup=False)
    proj.save()
        

