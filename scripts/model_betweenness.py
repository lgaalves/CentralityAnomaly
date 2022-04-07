#!/usr/bin/python

#Third Party Libraries
import networkx as nx
import igraph
import pandas as pd
import numpy as np

def net_name(init_nodes,final_nodes,m,delta,w0,rc):
    file='{}_{}_{}_{}_{}_{}'.format(init_nodes,final_nodes,m,round(delta,2),w0,round(rc,4))
    return file

def calculate_kbs_samples(method,file,nodes,n_samples):
    rk=[]
    rb=[]
    rs=[]
    for i in range(1,n_samples+1):
        edges=np.array(pd.read_csv('../spatial_model/{}/{}-{}.txt'.format(method,file,i),header=None).replace(np.nan,0))
        null_model=nx.Graph()   
        null_model.add_nodes_from(nodes)
        null_model.add_weighted_edges_from(edges)
        rdegree=nx.degree(null_model)
        rdegree=list(dict(rdegree).values())

        strength=nx.degree(null_model,weight='weight')
        strength=list(dict(strength).values())

        rbc=nx.betweenness_centrality(null_model,weight='weight',normalized=False)
        rbc=list(dict(rbc).values())


        rk.append(rdegree)
        rb.append(rbc)
        rs.append(strength)
    pd.DataFrame(rk).to_csv('../spatial_model/{}/k_{}.csv'.format(method,file))
    pd.DataFrame(rs).to_csv('../spatial_model/{}/s_{}.csv'.format(method,file))
    pd.DataFrame(rb).to_csv('../spatial_model/{}/b_{}.csv'.format(method,file))

def main():
    init_nodes = 5
    final_nodes = 1000
    m = 4
    w0=1
    n_samples=10000
    nodes=[i for i in range(1,final_nodes)]
    for rc in [0.01,1,10]:
        for delta in [0.01,1,10]:
            file=net_name(init_nodes,final_nodes,m,delta,w0,rc)
            print(file)
            calculate_kbs_samples(method='UBCM',file=file,nodes=nodes,n_samples=n_samples)
            calculate_kbs_samples(method='UECM',file=file,nodes=nodes,n_samples=n_samples)

if __name__ == "__main__":
    main()
