# Reproducibility Study
Scripts for manuscript *Pulmonary Ventilation Analysis Using UTE Lung MRI: A Reproducibility Study*

## Usage
1. phase resolved recon
    * main function: pr_recon_bash.sh, pr_recon.m
2. antspynet segmentation
    * main function: AntsSegmentation-Cyclic.py
3. constrained registration 
    * elastix.py: Elastix 3d+t registration
    * ants_syn.py: ANTs SyN registration
4. stats analysis
    * split_violin.py: split violin plot of ventilation
    * bland_altman.py: Bland-Altman plot of total ventilation
    * bland_altman_6zone.py: Bland-Altman plot of total ventilation in 6 zones
    * ants_intra_subject.py: register between two scans
    * coefficient_of_variation.py: coefficient of variation between two scans

## Dependency
  * [ANTsPy](https://antspy.readthedocs.io/en/latest/) and [ANTsPyNet](https://github.com/ANTsX/ANTsPyNet)
  * [SimpleITK](https://simpleitk.org) with [Elastix](http://simpleelastix.github.io)
  * [Seaborn](https://seaborn.pydata.org)
  * [SigPy](https://sigpy.readthedocs.io/en/latest/)
