function el=adj2edge(adj)
n=length(adj); % number of nodes
edges=find(adj>0); % indices of all edges
el=[];
for e=1:length(edges)
[i,j]=ind2sub([n,n],edges(e)); % node indices of edge e  
el=[el; i j adj(i,j)];
end