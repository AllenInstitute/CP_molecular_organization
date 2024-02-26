# Molecular determinants of input-output connectivity in mouse caudoputamen

This repository contains a set of python scripts for performing analyses on anterograde, retrograde and single neuron data used in Wang et al.

### Installation and dependencies

All analysis and visualization code has been written in Python3. Scripts require a number of dependencies, which are listed in the `caudoputamen.yml` file. Dependencies can be installed in a conda environment using:

`conda env create -f caudoputamen.yml`

#### Notes on usage

 - Analysis requires CCFv3 annotation volumes not included in this repository. Annotation volumes can be downloaded using the Allen SDK, instructions [here](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/reference_space.html#Downloading-an-annotation-volume).
 - Cortical flatmapping for Figs 3 and 4 is explained in depth [here](https://ccf-streamlines.readthedocs.io/en/latest/)
 - The data directory contains sample swcs for generating analyses and figures for Figure 4. 

### Level of support

We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.