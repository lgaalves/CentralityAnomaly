function A=CM_sampling(sol)
%function A=CM_sampling(sol)

%Procedure for drawing a matrix according to the Undirected Binary Configuration Model

n=length(sol);
P=zeros(n,n);
A=zeros(n,n);

for i = 1:n
    for j = (i+1):n
                      
        %Generating a binary matrix extracted froma a Bernoulli
        %distribution
        
        %The probability is given by the solution of the UBCM model
        
        P(i,j) =(sol(i)*sol(j))/(1 + (sol(i)*sol(j)));
                      
        %Random real number in the range (0,1) extracted 
        %from the standard uniform distribution
        
        z = rand(1);
        
        if z < P(i,j)
                          
            A(i,j) = 1;
            A(j,i) = 1;
        end
    end
end
        
       