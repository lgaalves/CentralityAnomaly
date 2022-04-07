#!/usr/bin/python

#Third Party Libraries
import networkx as nx
import numpy as np
import pandas as pd

def rand_prob_node(G,dni,rc):
    nodes_probs = []
    for node in G.nodes():
        swi = G.degree(node,weight='weight')
        #print(node_degr)
        node_proba = swi*np.exp(-dni[node]/rc)
        #print("Node proba is: {}".format(node_proba))
        nodes_probs.append(node_proba)
        #print("Nodes probablities: {}".format(nodes_probs))
    nodes_probs=np.array(nodes_probs)
    nodes_probs=nodes_probs/np.sum(nodes_probs)
    random_proba_node = np.random.choice(G.nodes(),p=nodes_probs)
    #print("Randomly selected node is: {}".format(random_proba_node))
    return random_proba_node

def add_edge(G,new_node,dni,w0,rc):
    random_proba_node = rand_prob_node(G,dni,rc)
    new_edge = (random_proba_node, new_node, w0)
    if G.has_edge(random_proba_node, new_node)==True:
        G[random_proba_node][new_node]['weight']=G[random_proba_node][ new_node]['weight']+w0
    else:
        G.add_weighted_edges_from([(new_node, random_proba_node,w0)])
    return G,random_proba_node

def update_weights(G,random_proba_node,delta):
    swi = G.degree(random_proba_node,weight='weight')
    for i in list(dict(G[random_proba_node]).keys()):
        wij=G[random_proba_node][i]['weight']
        G[random_proba_node][i]['weight']=wij+delta*wij/swi
    return G

def main():
    init_nodes = 5
    final_nodes = 1000
    m = 4
    w0=1
    rc=0.1
    delta=1
    for rc in [0.01,1,10]:
        for delta in [0.01,1,10]:
            distances=np.random.uniform(size=init_nodes) # random positions
            dni=dict()
            for node in range(0,init_nodes):
                dni[node]=distances[node]
            # print("Creating initial graph...")

            edges=[] # edges
            for i in range(0,init_nodes):
                for j in range(0,init_nodes):
                    if i > j:
                        edges.append((i,j,w0))
            G=nx.Graph()
            G.add_weighted_edges_from(edges)

            # print("Graph created. Number of nodes: {}".format(len(G.nodes())))
            # print("Adding nodes...")

            for new_node in range(init_nodes,final_nodes):
                dni[new_node]=np.random.uniform()
                G.add_node(new_node)
                for e in range(0, m):
                    G,random_proba_node=add_edge(G,new_node,dni,w0,rc)
                    G=update_weights(G,random_proba_node,delta)

            degree=nx.degree(G)
            degree=list(dict(degree).values())
            strength=nx.degree(G,weight='weight')
            strength=np.array(list(dict(nx.degree(G,weight='weight')).values()))
            bc=nx.betweenness_centrality(G,normalized=False)
            bc=list(dict(bc).values())
            kbs_df=pd.DataFrame(np.transpose([degree,strength,bc]))
            kbs_df.to_csv('../spatial_model/UBCM_ksb_{}_{}_{}_{}_{}_{}.txt'.format(init_nodes,final_nodes,m,round(delta,2),w0,round(rc,4)))

            bc=nx.betweenness_centrality(G,weight='weight',normalized=False)
            bc=list(dict(bc).values())
            kbs_df=pd.DataFrame(np.transpose([degree,strength,bc]))
            kbs_df.to_csv('../spatial_model/UECM_ksb_{}_{}_{}_{}_{}_{}.txt'.format(init_nodes,final_nodes,m,round(delta,2),w0,round(rc,4)))

            strength=strength.astype(int)
            degree.extend(strength)
            df=pd.DataFrame(np.transpose([degree]),columns=['k'])
            print('{}_{}_{}_{}_{}_{}'.format(init_nodes,final_nodes,m,round(delta,2),w0,round(rc,4)))
            df.to_csv('../spatial_model/{}_{}_{}_{}_{}_{}.txt'.format(init_nodes,final_nodes,m,round(delta,2),w0,round(rc,4)),
                  header=None,
                 sep='\t',
                 index=False)
            # print('Done')


if __name__ == "__main__":
    main()
