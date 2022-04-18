#!/usr/bin/python

#Third Party Libraries
import networkx as nx
import numpy as np
import pandas as pd
import math
import random

def select_node(nodes, probabilities):
    """
    Choose a node according to probability
    
    Parameters
    ----------
    nodes:  list of ints
        List of nodes
    probabilities: list of floats [0,1]
        Probability of connection to each node
   
    Returns
    ------- 
    item: int
        Selected node 
    """
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for node, node_probability in zip(nodes, probabilities):
        cumulative_probability += node_probability
        if x < cumulative_probability:
            break
    return int(node)

def generate_coordinates(r, pt_cnt):
    """
    
    Generates the coordinates of nodes randomly located on disc of radius r 
    following the uniform distributions.
    
    
    Parameters
    ----------
    r: float
        Radius of the disc
    pt_cnt: int
        Number of coordinates
    
    Returns
    -------
    coordinates: list
        List of coordinates for all nodes
    """
    x_list = []
    y_list = []
    for k in range(pt_cnt):
        random_theta = random.random() * math.pi * 2
        random_r = math.sqrt(random.random()) * r
        x = math.cos(random_theta) * random_r
        y = math.sin(random_theta) * random_r
        x_list.append(x)
        y_list.append(y)
    coordinates = [x_list, y_list]
    return coordinates

def euclidean_distance(a,b):
    """
    Compute Euclidean distance between two a and b
    
    Parameters
    ----------
    a: (float, float) 
        First coordinate
    b: (float, float) 
        Second coordinate
        
    Returns
    ----------
    
    distance: float
        Euclidean distance between a and b
    
    """
    distance = np.linalg.norm(a-b)
    return distance

def strength(G):
    """
    Compute strength of nodes
    
    Parameters
    ----------

    G: Networkx graph object
        Model network
        
    Returns
    ----------
    nodes: list of ints
        Nodes indices
    s: list of floats
        Strength of nodes
    """
    nodes,s = np.transpose(G.degree(weight='weight'))
    return list(nodes), list(s)

def compute_probabilities(s,distances,rc):
    """
    Compute probabilities based on distances
    
    Parameters
    ----------
    s: list of floats
        Strength of nodes
    distance: list of floats
        Euclidean distance between a and b
    rc: float
        Typical spatial scale
        
    Returns
    ----------
    
    probabilities: list of floats [0,1]
        Probability of connection to each node
    """
    probabilities = np.multiply(np.array(s),np.exp(-1 * distances / rc))
    if probabilities.sum()>0:
        probabilities = list(probabilities / probabilities.sum()) 
    else:
        probabilities = [1/len(probabilities) for i in range(0,len(probabilities))]
    return probabilities

def initialize_graph(M0,w0=1):
    """
    Generate the complete graph
 
    Parameters
    ----------
    M0: int
        Number of initial nodes
    w0: float
        Initial weight of edges
        
    Returns
    ----------
    G: Networkx graph object
        Model network
    """
    G = nx.complete_graph(M0, create_using=None)
    
    for edge in G.edges():
        G.add_edge(edge[0], edge[1], weight=w0)
        
    return G

def select_m_nodes(nodes,probabilities,m):
    """
    Select m different nodes following the linking probability 
    
    Parameters
    ----------    
    nodes: list of ints
        Nodes indices
    probabilities: list of floats [0,1]
        Probability of connection to each node

    Returns
    ---------- 
    selected_nodes: list of ints
        List of selected nodes
    """
    selected_nodes = []
    node = select_node(nodes, probabilities)
    while (len(selected_nodes) < m):
        if node in selected_nodes:
            node = select_node(nodes, probabilities)
        else:
            selected_nodes.append(node)
    return selected_nodes

def update_weights(G,node,delta,s):
    """
    Update the link weights
    
    Parameters
    ---------- 
    G: Networkx graph object
        Model network
        
    node: int
        Node label
    delta: float
        Susceptibility of the network to new links
    s: list of floats
        Strength of nodes    
    Returns
    ---------- 
    G: Networkx graph object
        Model network
    s: list of floats
        Strength of nodes
    """
    
    for neighbor in G.neighbors(node):
        wij = G.edges[(neighbor, node)]['weight']
        G.add_edge(neighbor, node, weight= wij + (delta * (wij / s[node])))
        s[node] = G.degree(node, weight='weight')
        s[neighbor] = G.degree(neighbor, weight='weight')
    return G,s

def add_new_node(m, L, rc, w0, delta, G, x, y ,new_node):
    """
    Parameters
    ----------       
    m: int
        Number of links added at each time step
    L: float
        Radius of two-dimensional disc
    rc: float
        Typical spatial scale
    w0: float
        Initial weight of edges
    delta: float
        Susceptibility of the network to new links
    G: Networkx graph object
        Model network
    x: list of floats
        Initial x-positions
    y: list of floats
        Initial y-positions
    new_node: int
        Label of new node
    
    Returns
    --------
    """
  
    # Get old positions
    pos_old = np.array([x, y])
    
    # Generates the coordinates for the new node
    [new_x, new_y] = generate_coordinates(L, 1)
    pos_new = np.array([new_x, new_y])
    
    # Append the coordinate of new node to the list of old nodes
    x.append(new_x[0])
    y.append(new_y[0])
    
    # Calculate the Euclidean distances between new node and all old nodes
    distances = euclidean_distance(pos_old, pos_new)
    
    # calculate the strength of all nodes
    nodes, s = strength(G)
    
    # Compute probabilities
    probabilities = compute_probabilities(s,distances,rc)
    
    # Select m different old nodes following the linking probability
    selected_nodes = select_m_nodes(nodes,probabilities,m)

    for t,node in enumerate(selected_nodes):
        # add links between new node and selected nodes
        G.add_edge(node,new_node, weight=w0)
        
        # update the strength of old nodes
        s[node] = G.degree(node, weight='weight')
        
        # append or update the strength of new node
        if t == 0:
            s.append(G.degree(new_node, weight='weight'))
        else:
            s[new_node] = G.degree(new_node, weight='weight')
            
        # Update weights   
        G,s = update_weights(G,node,delta,s)
        
    return G,x,y

def SDPASS(N, M0, m, L, rc, w0, delta):
    """
    Generate the network according to SDPASS model
    
    Parameters
    ----------
    
    N: int
        Size of networks
    M0: int
        Number of initial nodes
    m: int
        Number of links added at each time step
    L: float
        Radius of two-dimensional disc
    rc: float
        Typical spatial scale
    w0: float
        Initial weight of edges
    delta: float
        Susceptibility of the network to new links
         
    Returns
    ----------
    G: Networkx graph object
        Model network
    """
    
    # Create complete graph with M0 nodes
    G = initialize_graph(M0,w0=1)

    # Generates the coordinates of M0 nodes
    [x, y] = generate_coordinates(L, M0)
    
    # add a new node by the loop
    for new_node in range(M0, N):
        G,x,y = add_new_node(m, L, rc, w0, delta, G, x, y ,new_node)
    return G

def main():
    init_nodes = 5
    final_nodes = 100
    m = 4
    w0=1
    rc=0.1
    delta=1
    for rc in [0.01,1,10]:
        for delta in [0.01,1,10]:
            G = SDPASS(N=final_nodes, M0=init_nodes, m=m, L=1, rc=rc, w0=w0, delta=delta)

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
