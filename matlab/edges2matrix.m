function W=edges2matrix(List)
% function W=edges2matrix(List)

% Procedure for transofrming the data in edge-list form in matrix-form 

v_id_from = List(:,1);
v_id_to = List(:,2);
v_id_all = [v_id_from; v_id_to];

% Determining non-repeated codes

%NB: the code is in an increasing order

v_id_unique = unique(v_id_all);

%Associating to each couple of codes (sorce/target)the proper new position
%according to the vector "v_id_unique"

[~,from] = ismember(v_id_from, v_id_unique);
[~,to] = ismember(v_id_to, v_id_unique);

%Using the code in "from" and "to" and the weights in "List(:,3) to fill
%the adjancency matrix "A"

l=length(v_id_unique);
W=zeros(l,l);

for i=1:length(from)
    W(from(i),to(i))=List(i,3);
end