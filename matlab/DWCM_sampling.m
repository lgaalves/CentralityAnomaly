function W=DWCM_sampling(sol) 
%function W=DWCM_sampling(sol) 

%Procedure for drawing a matrix according to the Directed Weighted Configuration Model

n=length(sol);
m = n/2;
x = sol(1:m);
y = sol((m+1):end);
P = zeros(m,m);
W=zeros(m,m);        

for i = 1:m
    for j = 1:m
        if i ~= j
            
            %Generating a weighted matrix extracted froma a geometric
            %distribution with given probability (1-P)
                                                  
            %The probability is given by the solution of the DWCM model
            
            P(i,j) = x(i)*y(j);
            
            t = geornd((1 - P(i,j)));  
            W(i,j) = t;
        end
    end
end
     
        
        