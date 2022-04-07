function W=WCM_sampling(sol) 
%function W=WCM_sampling(sol)

%Procedure for drawing a matrix according to the Undirected Weighted Configuration Model

n=length(sol);
P=zeros(n,n);
W=zeros(n,n);

for i = 1:n
    for j = (i+1):n
                  
        %Generating a weighted matrix extracted froma a geometric
        %distribution with given probability (1-P)
                                                  
        %The probability is given by the solution of the UWCM model
        
        P(i,j) =sol(i)*sol(j);
                      
        t = geornd((1 - P(i,j)));  
                                                  
        W(i,j) = t;
        W(j,i) = W(i,j);
                     
    end
end