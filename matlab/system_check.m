function out = system_check(method,sol,M,arg)
%function out = system_check(method,sol,M,arg)

% Procedure for:

% 1) solving the system associated to the maximization entropy problem for
%   all models ("arg" = 0);
%
% 2) checking if the constraint(s) is(are) satisfied and deciding when the
%    required precision is reached ("arg" = 1) 

n=length(sol);

switch upper(method)
    
    case 'UBCM' 
   
       %Compute the expected values for the constraints 
        
        Exp=zeros(n,1);
        for i = 1:n
            for j = (i+1):n
                P(i,j) =(sol(i)*sol(j))/(1 + (sol(i)*sol(j)));
                P(j,i) = P(i,j);
            end
        end
        
        Exp=sum(P)';
        
        if size(M,2) >1 %(we are in the matrix case so we need to compute 
                        % the observed values for the constraints) 
           
            Obs = sum(M)';
        else if size(M,2)==1
            Obs = M;
            end
        end
        
       
    case 'UWCM' 
        
        %Compute the expected values for the constraints 
        
        S=zeros(n,n);
        for i=1:n
            for j=(i+1):n
                S(i,j) = (sol(i)*sol(j))/ (1-(sol(i)*sol(j))) ;
                S(j,i) = S(i,j);
            end
        end
        
        Exp = sum(S)';
        
        if size(M,2) >1 %(we are in the matrix case so we need to compute 
                        % the observed values for the constraints) 
          
            Obs = sum(M)';
        else
            Obs = M;
        end
    
    case 'DBCM' 
        
        m = n/2;
        x = sol(1:m);
        y = sol((m+1):end);
        
        %Compute the expected values for the constraints 
        
        P = zeros(m,m);
        
        for i = 1:m
            for j = 1:m
                  if i ~= j
                    P(i,j) = (x(i)*y(j))/(1 + (x(i)*y(j)));
                      
                  end
            end
        end
        
        Exp=[sum(P'),sum(P)]';
        
        %Compute the observed values for the constraints  
        
         if size(M,2) >1 %(we are in the matrix case so we need to compute 
                        % the observed values for the constraints) 
           
             Obs=[sum(M'),sum(M)]';
         else
             Obs = M;
         end
      
    case 'DWCM'
        
        m = n/2;
        x = sol(1:m);
        y = sol((m+1):end);
        
        %Compute the expected values for the constraints 
        P = zeros(m,m);
        
        for i = 1:m
            for j = 1:m
                  if i ~= j
                    P(i,j) = (x(i)*y(j))/(1-(x(i)*y(j)));
                      
                  end
            end
        end
        
        Exp=[sum(P'),sum(P)]';
        
        if size(M,2) >1  %(we are in the matrix case so we need to compute 
                         % the observed values for the constraints) 
          
            Obs=[sum(M'),sum(M)]';
        else 
            Obs = M;
        end
        
    case 'UECM' 
            
         m = n/2;
         x = sol(1:m);
         y = sol((m+1):end);   
         
         %Compute the expected values for the constraints 
         
         P=zeros(m,m);
         S=zeros(m,m);
         
         for i=1:m
             for j=(i+1):m
                 P(i,j)=(x(i)*x(j)*y(i)*y(j))/(1-y(i)*y(j)+x(i)*x(j)*y(i)*y(j));
                 P(j,i)=P(i,j);
                 
                 S(i,j)=P(i,j)/(1-y(i)*y(j));
                 S(j,i)=S(i,j);
             end
         end
         
         Exp=[sum(P),sum(S)]';
         
         if size(M,2) >1  %(we are in the matrix case so we need to compute 
                        % the observed values for the constraints) 
             
             A=fix(M>0);
             Obs=[sum(A),sum(M)]';
         else
             Obs = M;
         end
         
    case 'RBCM' 
        
        for l=1:n
            if sol(l) <10^(-3)
                sol(l)=0;
            end
        end
            
         q = n/3;
         x = sol(1:q);
         y = sol((q+1):2*q);
         z=sol((2*q+1):end);
         
         %Compute the expected values for the constraints 
         
         pout=zeros(q,q);
         pin=zeros(q,q);
         prec=zeros(q,q);
         
         for i=1:q
             for j=1:q
                 if i~=j
                     
                     d=1+x(i)*y(j)+x(j)*y(i)+z(i)*z(j);
                     pout(i,j)=(x(i)*y(j))/d;
                     pin(i,j)=(x(j)*y(i))/d;
                     prec(i,j)=(z(i)*z(j))/d;
                     
                 end
             end
         end
         
         Exp=[sum(pin),sum(pout),sum(prec)]';
           
        if size(M,2) >1  %(we are in the matrix case so we need to compute 
                        % the observed values for the constraints) 
         
            [Aout,Ain,Arec]=rec(M);
            Obs=[sum(Ain),sum(Aout),sum(Arec)]';
        else
            Obs = M;
        end
       
    case 'RWCM' 
        
        for l=1:n
            if sol(l) <10^(-3)
                sol(l)=0;
            end
        end
        
        q = n/3;
        
        x = sol(1:q);
        y = sol((q+1):2*q);
        z=sol((2*q+1):end);
         
        %Compute the expected values for the constraints  
        
        W_out_exp=zeros(q,q);
        W_in_exp=zeros(q,q);
        W_rec_exp=zeros(q,q);
         
        for i=1:q
            for j=1:q
                if i ~= j
                    W_out_exp(i,j) = ((x(i)*y(j))*(1-(x(j)*y(i))))/((1-(x(i)*y(j)))*(1-(x(i)*x(j)*y(i)*y(j))));
                    W_in_exp(i,j)  = ((x(j)*y(i))*(1-(x(i)*y(j))))/((1-(x(j)*y(i)))*(1-(x(i)*x(j)*y(i)*y(j))));
                    W_rec_exp(i,j) = (z(i)*z(j))/(1-(z(i)*z(j)));
                end
            end
        end
        
        Exp=[sum(W_in_exp),sum(W_out_exp),sum(W_rec_exp)]';
       
        if size(M,2) >1  %(we are in the matrix case so we need to compute 
                        % the observed values for the constraints) 
            
            [Wout,Win,Wrec]=recW(M);
            Obs=[sum(Win),sum(Wout),sum(Wrec)]';
        else
          
            Obs = M;
        end
end

if arg == 0
       
out = Exp - Obs;
else
    diff = abs(Obs - Exp);
    
    for i=1:length(diff)
        if Obs(i)==0
            
            temp(i)=diff(i);
        else
            
            temp(i)= diff(i)/Obs(i);
        end
    end
    out=max(temp);
end
        
    
            
            
            

