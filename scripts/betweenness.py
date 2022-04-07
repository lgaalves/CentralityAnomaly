#!/usr/bin/python

#Third Party Libraries
import networkx as nx
import igraph
import pandas as pd
import numpy as np

def read_samples(net,number_of_samples=1):
    sample=[]
    for i in range(1,number_of_samples+1):
        df=pd.read_csv('../samples/Networks/{}/random_{}.txt'.format(net,i),sep=' ',
            names=['Source','Target','Weight'],index_col=None)
        df.Source=df.Source.astype(int)-1
        df.Target=df.Target.astype(int)-1
        sample.append(df)
    return sample

def read_real_data(net):
    df=pd.read_csv('../samples/network_{}.txt'.format(net),sep=' ',header=0,index_col=None)
    df.Source=df.Source.astype(int)-1
    df.Target=df.Target.astype(int)-1
    return df 

def nonweighted_ksb(df,nodes):
    g=igraph.Graph()
    g.add_vertices(nodes)
    g.add_edges(np.array(df[['Source','Target']]))
    b=g.betweenness()
    k=g.degree()
    s=g.strength()
    return k,s,b

def weighted_ksb(df,nodes):
    g=igraph.Graph(directed=False)
    g.add_vertices(nodes)
    g.add_edges(np.array(df[['Source','Target']]))
    b=g.betweenness(weights=1/np.array(df.Weight))
    k=g.degree()
    s=g.strength(weights=np.array(df.Weight))
    return k,s,b

def calculate_ksb_for_samples(df_real,sample,weighted):
    ksample=[]
    ssample=[]
    bsample=[]
    nmax=max(set(df_real.Source).union(set(df_real.Target)))
    nodes=[i for i in range(0,nmax+1)]
    for item in range(0,len(sample)):
        if weighted==False:
            k,s,b=nonweighted_ksb(sample[item],nodes=nodes)
        else:
            k,s,b=weighted_ksb(sample[item],nodes=nodes)
        ksample.append(k)
        ssample.append(s)
        bsample.append(b)
    return ksample,ssample,bsample


def calculate_and_save(net,number_of_samples):
    df_real=read_real_data('{}_weight'.format(net))

    # UBCM - non-weighted method
    ubcm_sample=read_samples('{}'.format(net),number_of_samples=number_of_samples)
    ksample,ssample,bsample=calculate_ksb_for_samples(df_real,ubcm_sample,weighted=False)
    pd.DataFrame(ksample).to_csv('../results/kbs/{}_k.csv'.format(net))
    pd.DataFrame(ssample).to_csv('../results/kbs/{}_s.csv'.format(net))
    pd.DataFrame(bsample).to_csv('../results/kbs/{}_b.csv'.format(net))

    # UECM - non-weighted method
    df=df_real.copy()
    numberofnodes=max(np.array(list(set(df_real.Source).union(set(df_real.Target)))))
    df.Source=df.Source
    df.Target=df.Target
    kw,sw,bw=weighted_ksb(df,nodes=[i for i in range(0,numberofnodes+1)])

    data=read_samples('{}_weight'.format(net),number_of_samples=number_of_samples)
    uecm_sample=[]
    for i in range(0,number_of_samples):
        df=data[i]
        df['Weight']=df['Weight']*(min(sw))
        uecm_sample.append(df)

    wksample,wssample,wbsample=calculate_ksb_for_samples(df_real,uecm_sample,weighted=True)
    pd.DataFrame(wksample).to_csv('../results/kbs/{}_weight_k.csv'.format(net))
    pd.DataFrame(wssample).to_csv('../results/kbs/{}_weight_s.csv'.format(net))
    pd.DataFrame(wbsample).to_csv('../results/kbs/{}_weight_b.csv'.format(net))

def main():
    for net in ['UK','ES','BR','AIR']:
        print(net)
        calculate_and_save(net,number_of_samples=10000)

if __name__ == "__main__":
    main()
