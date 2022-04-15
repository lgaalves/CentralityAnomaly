# Centrality anomalies in complex networks as a result of model over-simplification

### What is this repository for? ###

This repository contains the code for the analysis reported in [New Journal of Physics 23, 013043](https://lgaalves.github.io/publications/2020-njp-anomaly-detection.pdf).

<image src='featured.png' />


### Disclaimer 

Please note that this repository is not intended for wide-spread distribution. We are only making the code available so that other researchers may reproduce the results published in our manuscript. 

### Directory Structure ###

* [data](/data) -- real networks data
* [figures/raw_figures](/figures/raw_figures) -- folder to save 
* [matlab](/matlab) -- code to generate samples 
* [notebooks](/notebooks) -- contain the notebook to generate the figures
* [python](/python) -- for python modules/functions
* [results/kbs](/results/kbs) -- folder to save degree, strengh, and betweenness centrality
* [samples](/samples) -- store network ensembles
* [scripts](/scripts) -- python code to run on terminal 
* [spatial_model](/spatial_model) -- folder with results for spatial model

## How to use

```bash
$conda create -n anomaly
$conda activate anomaly
$conda config --env --add channels conda-forge
$conda config --env --set channel_priority strict
$conda install python=3.6 networkx=2.4 matplotlib=3.0.3 pandas cartopy geopandas astroml seaborn python-igraph jupyter
$pip install watermark
```

See further explanations here:

* [Generate the ensembles](/matlab/README.md)
* [Generate SDPASS model data](/scripts/README.md)
* [Generate the figures](/notebooks/anomaly.ipynb)

# Reference

Luiz G. A. Alves, Alberto Aleta, Francisco A. Rodrigues, Yamir Moreno, Luís A. N. Amaral, Centrality anomalies in complex networks as a result of model over-simplification. [New Journal of Physics 23, 013043](https://lgaalves.github.io/publications/2020-njp-anomaly-detection.pdf)