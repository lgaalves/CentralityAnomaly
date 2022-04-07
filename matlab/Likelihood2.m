function f = Likelihood2(h,Par,method)
%function f = Likelihood2(h,Par,method)


%Procedure for computing the likelihood function for each model when the
%input are the constraints

n = length(h);
f = 0;

switch upper(method)

    case 'UBCM' % Undirected Binary Configuration Model
        
        % k --> degree
        
        k=Par;
        
        for i = 1:n
            
            f = f + log(h(i)^k(i));
            
            for j = (i+1):n
                f = f - log(1 + h(i)*h(j));
            end
        end
       
    case 'UWCM'   % Undirected Weighted Configuration Model
        
        % s --> strength
        
        s=Par;
        
        for i=1:n
            
            if  h(i)==0
               f = f + log(h(i)^s(i)); 
            else
                f = f + s(i)*log(h(i));
            end
                
            for j=(i+1):n
                 f = f + log(1 - h(i)*h(j));
            end
        end
        
       
    case 'DBCM' % Directed Binary Configuration Model
        
        m=n/2;
        x=h(1:m);
        y=h((m+1):end);
        
        %ko --> k_out
        %ki --> k_in
        
        ko=Par(1:m);
        ki=Par((m+1):end);
        
        for i = 1:m 
            
            f = f + log(x(i)^ko(i)) + log(y(i)^ki(i));
  
            for j = 1:m
                if i ~= j
                    f = f - log(1 + x(i)*y(j));
                end
            end
        end
        
    case 'DWCM'  % Directed Weighted Configuration Model
        
        
        m=n/2;
        x=h(1:m);
        y=h((m+1):end);
        
        %so --> s_out
        %si --> s_in
        
        so=Par(1:m);
        si=Par((m+1):end);
        
        
       for i = 1:m 
              
           f = f + so(i)*log(x(i)) + si(i)*log(y(i)); 
           
           for j = 1:m
               if i ~= j
                   f = f + log(1 - x(i)*y(j));
               end
           end
       end
        
        
    case 'UECM'  % Undirected Enhanced Configuration Model
        
        m=n/2;
        x=h(1:m);
        y=h((m+1):end);
        
        %k --> degree
        %s --> strength
        
        k=Par(1:m);
        s=Par((m+1):end);
      
        for i=1:m
            if h(i)==0
                f = f + log(x(i)^k(i)) + log(y(i)^s(i));
            else
                f = f + k(i)*log(x(i)) + s(i)*log(y(i));
            end
            for j=(i+1):m
                f = f + log(1-y(i)*y(j))-log(1-(y(i)*y(j))+x(i)*x(j)*y(i)*y(j));
            end
        end
        
    case 'RBCM'   % Reciprocal Bianry Configuration Model
       
        q=n/3;
        x=h(1:q);
        y=h((q+1):2*q);
        z=h((2*q+1):end);
        
        %ko --> k out only
        %ki --> k in only
        %kr --> k rec
        
        ko=Par(1:q);
        ki=Par((q+1):2*q);
        kr=Par(((2*q)+1):end);
        
        for i=1:q
            f = f + log(x(i)^ko(i)) + log(y(i)^ki(i)) + log(z(i)^kr(i));
            for j=(i+1):q
                f = f - (log(1 + x(i)*y(j) + x(j)*y(i) + z(i)*z(j)));
            end
        end
        
    case 'RWCM'  %Reciprocal Weighted Configuration Model
        
        q=n/3;
        x=h(1:q);
        y=h((q+1):2*q);
        z=h((2*q+1):end);
        
        %so --> s out only
        %si --> s in only
        %sr --> s rec
        
        so=Par(1:q);
        si=Par((q+1):2*q);
        sr=Par(((2*q)+1):end);
        
        for i=1:q
            
            f = f + so(i)*log(x(i)) + si(i)*log(y(i)) + sr(i)*log(z(i));
          
            for j=(i+1):q
                f = f + (log(1 - x(i)*y(j)) +log(1 - x(j)*y(i)) + log(1 - z(i)*z(j)) - log(1 - x(i)*y(j)*x(j)*y(i)));
            end
        end
        
end

f=-f;
