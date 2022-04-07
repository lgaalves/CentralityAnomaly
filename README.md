# README #

### What is this repository for? ###

* Analyzing data from buses


### Directory Structure ###

* python -- for python scripts
* python/modules -- for python modules
* data -- for datasets that might be helpful to add to repo
* docs -- for documentation and notes
* results -- for figures or summary files


## Installation

conda create -n anomaly      
conda activate anomaly
conda config --env --add channels conda-forge  
conda config --env --set channel_priority strict
conda install python=3.6 networkx=2.4 matplotlib=3.0.3 pandas cartopy  geopandas astroml seaborn python-igraph jupyter
pip install watermark