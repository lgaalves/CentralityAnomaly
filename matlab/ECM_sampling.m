function W=ECM_sampling(sol)
%function W=ECM_sampling(sol)

%Procedure for drawing a matrix according to the Undirected Enhanced Configuration Model

n = length(sol);
m = n/2;
x = sol(1:m);
y = sol((m+1):end);
P=zeros(m,m);
W=zeros(m,m);

for i = 1:m
    for j =(i+1):m
      
            %Step 1
            
            %Generating a binary matrix extracted froma a Bernoulli
            %distribution
            
            %Step 2
            
            %Generating a weighted matrix extracted from a geometric
            %distribution with given probability (1-P) and using the binary
            %structur in step 1)
                                                  
            %The probabilities are given by the solution of the UMCM model
            
            P(i,j)  = (x(i)*x(j)*y(i)*y(j))/(1 - y(i)*y(j) + x(i)*x(j)*y(i)*y(j));
            PY(i,j) = y(i)*y(j);
        
            %Random real number in the range (0,1) extracted 
            %from the standard uniform distribution
            
            z = rand(1);   
            
            if z < P(i,j) 
                
                A(i,j) = 1;
                A(j,i) = 1;
                
                %Integer number extracted from a geometric
                %distribution with given probability (1-PY)
                
                t = geornd((1 - PY(i,j)));  
                
                W(i,j) = 1 + t;
                W(j,i) = W(i,j);
            end
        end
    end
end



    
            
            
