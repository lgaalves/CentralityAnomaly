# Unbiased sampling of network ensembles

Code adapted from paper by Tiziano Squartini, Rossana Mastrandrea, and Diego Garlaschelli (2015), Unbiased sampling of network ensembles. New J. Phys. 17 023052. 

The original package was originally published by here: [A Matlab package to randomize and sample networks by a max-entropy approach for several null models
](https://www.mathworks.com/matlabcentral/fileexchange/46912-max-sam-package-zip)

# How to use it:

Generating ensemble using the MATLAB script:


### Undirected Binary Configuration Model (UBCM)

```
MATLAB >> pairks=load(strcat('network_degree_sequence.txt')))
MATLAB >> outputs = MAXandSAM('UBCM',[],pairks,[],10^(-6),0);
MATLAB >> path = '/path_to_samples_folder/network_name-'
MATLAB >> for i=1:num
........     W_ext=samplingAll(outputs,'UBCM');
........     edges=adj2edge(W_ext);
........     name=strcat(path,num2str(i));
........     dlmwrite(strcat(name,'.txt'),edges)
........  end
```

### Undirected Enhanced Configuration Model (UECM)

```
MATLAB >> pairks=load(strcat('network_degree_and_strenght_sequence.txt')))
MATLAB >> outputs = MAXandSAM('UECM',[],pairks,[],10^(-6),0);
MATLAB >> path = '/path_to_samples_folder/network_name-'
MATLAB >> for i=1:num
........     W_ext=samplingAll(outputs,'UECM');
........     edges=adj2edge(W_ext);
........     name=strcat(path,num2str(i));
........     dlmwrite(strcat(name,'.txt'),edges)
........  end
```

### Ensemble for SDPASS models

```
MATLAB >> SaveSpatialModelSamples('network_name.txt',1000) 
```

# Unbiased sampling of network ensembles in Python

Since we published the paper, it was released a version of the MATLAB package for Python. Please refer to the repository 
[NEMtropy: Network Entropy Maximization, a Toolbox Running On PYthon](https://github.com/nicoloval/NEMtropy
) for the Python code that generates the ensembles. 

# References
* Tiziano Squartini, Rossana Mastrandrea, and Diego Garlaschelli (2015), Unbiased sampling of network ensembles. [New J. Phys. 17 023052](https://iopscience.iop.org/article/10.1088/1367-2630/17/2/023052#njp509023app1). 
* Rossana (2022). [MAX&SAM package.zip](https://www.mathworks.com/matlabcentral/fileexchange/46912-max-sam-package-zip), MATLAB Central File Exchange. Retrieved April 7, 2022.
* Nicolò Vallarano,  Matteo Bruno,  Emiliano Marchese, Giuseppe Trapani, Fabio Saracco, Tiziano Squartini, Giulio Cimini, and Mario Zanon (2021) Fast and scalable likelihood maximization for Exponential Random Graph Models. [Arxiv:2101.12625](https://arxiv.org/abs/2101.12625)
