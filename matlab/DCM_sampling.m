function A=DCM_sampling(sol) 
%function A=DCM_sampling(sol) 

%Procedure for drawing a matrix according to the Directed Binary Configuration Model

n=length(sol);
m = n/2;

x = sol(1:m);
y = sol((m+1):end);
        
P = zeros(m,m);
A=zeros(m,m);       

for i = 1:m
    for j = 1:m
        if i ~= j
            
            %Generating a binary matrix extracted froma a Bernoulli
            %distribution
            
            %The probability is given by the solution of the DBCM model
            
            P(i,j) = (x(i)*y(j))/(1 + (x(i)*y(j)));
           
            %Random real number in the range (0,1) extracted 
            %from the standard uniform distribution
            
            z=rand(1);
                    
            if z < P(i,j)
                        
                A(i,j)=1;
            end
        end
    end
end