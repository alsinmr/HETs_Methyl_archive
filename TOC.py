#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:26:05 2023

@author: albertsmith
"""

import pyDR
import os



md_dir='/Volumes/My Book/HETs/MDSimulation'
topo='HETs_3chain.pdb'
traj=os.path.join(md_dir,'HETs_MET_4pw.xtc')


proj=pyDR.Project()

sys=pyDR.MolSys(topo=topo,traj_files=traj,project=proj)

proj.chimera.close()
sys.chimera()
proj.chimera.command_line(['~ribbon ~/B','show /B'])

sys.make_pdb(ti=100000)
sys.chimera()

sys.make_pdb(ti=200000)
sys.chimera()

sys.make_pdb(ti=300000)
sys.chimera()

proj.chimera.command_line(['~ribbon ~#1','show /B','transparency ~#1 70 target a',                        
                           'color #1 dark green','color #2 dark cyan','color #3 dodger blue','color #4 powder blue',
                           'color #1 light slate grey target r'])

for k in range(2,4):
    proj.chimera.command_line(f'align #{k}/B@C,CA,N toAtoms #1/B@C,CA,N')