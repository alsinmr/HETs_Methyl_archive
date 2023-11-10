# HETs_Methyl_archive
Archive of scripts used for HETs dynamics analysis


This archive provides python scripts and partially processed data to produce in the paper

"An Analytical Framework to Understand the Origins of Methyl Sidechain Dynamics in Protein Assemblies"

by

Kai Zumpfe, Melanie Berbon, Birgit Habenstein, Antoine Loquet, and Albert A. Smith

There is NO INSTALLATION required for the provided code. Just place everything in a folder, navigate there, and run with python3. However, python3 and the following modules must be installed from other sources (these are the tested versions, although other versions may work).

Python v. 3.7.3, numpy v. 1.19.2, scipy v. 1.5.2, MDAnalysis v. 1.0.0, matplotlib v. 3.4.2

Additionally, ChimeraX must be installed and its location provided to pyDR. How to do this is provided at a comment at the top of the provided scripts.

We recommend installing Anaconda: https://docs.continuum.io/anaconda/install/ The Anaconda installation includes Python, numpy, scipy, pandas, and matplotlib. (I also highly recommend using Spyder, which comes with Anaconda, or some other variant of iPython for running the provided scripts interactively, such that one may stop to understand each step in the overall analysis)

MDAnalysis is installed by running: conda config --add channels conda-forge conda install mdanalysis (https://www.mdanalysis.org/pages/installation_quick_start/)

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Copyright 2023 Albert Smith-Penzel

This research was supported by Deutsche Forschungsgemeinschaft (DFG) through DFG grant 450148812 (A. A. Smith), ESF grant SAB 100382164 (Kai Zumpfe, A. A. Smith), and through the CNRS.
