# Centrality anomalies in complex networks as a result of model over-simplification

### What is this repository for? ###

This repository contains the code for the analysis reported in [New Journal of Physics 23, 013043](https://lgaalves.github.io/publications/2020-njp-anomaly-detection.pdf).

<image src='featured.png' />


Please note that this repository is not intended for wide-spread distribution. We are only making the code available so that other researchers may reproduce the results published in our manuscript. 

### Directory Structure ###

* python -- for python modules/functions
* matlab -- code to generate samples 
* notebooks -- contain the notebook to generate the figures
* data -- for datasets that might be helpful to add to repo
* samples -- contain the data to generate the networks 


## Installation

```bash
$conda create -n anomaly
$conda activate anomaly
$conda config --env --add channels conda-forge
$conda config --env --set channel_priority strict
$conda install python=3.6 networkx=2.4 matplotlib=3.0.3 pandas cartopy geopandas astroml seaborn python-igraph jupyter
$pip install watermark
```

# Reference

Luiz G. A. Alves, Alberto Aleta, Francisco A. Rodrigues, Yamir Moreno, Luís A. N. Amaral, Centrality anomalies in complex networks as a result of model over-simplification. [New Journal of Physics 23, 013043](https://lgaalves.github.io/publications/2020-njp-anomaly-detection.pdf)