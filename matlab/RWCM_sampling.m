function W_ext=RWCM_sampling(sol)
%function W_ext=RWCM_sampling(sol)

%Procedure for drawing a matrix according to the Reciprocal Weighted Configuration Model

 n=length(sol);
 q = n/3;
 x = sol(1:q);
 y = sol((q+1):2*q);
 z=sol((2*q+1):end);
 
 W_temp_rec=zeros(q,q);

    for i=1:q
        for j=(i+1):q
            w = geornd((1 - z(i)*z(j)));  
            W_temp_rec(i,j) =w;
            W_temp_rec(j,i)=w;
        end
    end
    
    % 2) Three-events distribution according to the prossible events:
    %    (0,0) -> absence of non-reciprocal links;
    %    (1,0) -> presence of non-reciprocal out-links;
    %    (0,1) -> presence of non-reciprocal in-links.

    % The related probabilities are computed using the solutions of the
    % Reciprocal Weighted Configuration Model

    P1=zeros(q,q);
    P2=zeros(q,q);
    

    for i=1:q
        for j=1:q
            if i ~= j
                d = (1 -(x(i)*x(j)*y(i)*y(j)));
                P1(i,j)= ((1-x(i)*y(j))*(1-x(j)*y(i)))/d;
                P2(i,j)= ((x(i)*y(j))*(1-x(j)*y(i)))/d;
            end
        end
    end
    
    A_out_temp=zeros(q,q);
    A_in_temp=zeros(q,q);
    
    for i=1:q
        for j=(i+1):q
        
            int=[0,P1(i,j),P1(i,j)+P2(i,j),1];
            k=rand(1);
            f=is_in_class(k,int');
            
            if f==1
                A_out_temp(i,j)=0;
                A_out_temp(j,i)=0;
                A_in_temp(i,j)=0;
                A_in_temp(j,i)=0;
            else if f==2
                    A_out_temp(i,j)=1;
                    A_out_temp(j,i)=0;
                    A_in_temp(i,j)=0;
                    A_in_temp(j,i)=1;
                else if f==3
                        A_out_temp(i,j)=0;
                        A_out_temp(j,i)=1;
                        A_in_temp(i,j)=1;
                        A_in_temp(j,i)=0;
                    end
                end
            end
        end
    end
    
    % 3) After having defined the structure of non-reciprocal links, 
    % the weights, extracted from a geometric distribution with given
    % probabilities, are assigned to them.

    W_out_temp=zeros(q,q);
    W_in_temp=zeros(q,q);

    W_out_ext=zeros(q,q);
    W_in_ext=zeros(q,q);

    W_ext=zeros(q,q);

    for i=1:q
        for j=(i+1):q
            t1=geornd(1-x(i)*y(j));
            t2=geornd(1-x(j)*y(i));
            
            W_out_temp(i,j)=1+t1;
            W_in_temp(j,i)=1+t1;
            W_in_temp(i,j)=1+t2;
            W_out_temp(j,i)=1+t2;
        end
    end
    
    W_out_ext=A_out_temp.*W_out_temp;
    W_in_ext=A_in_temp.*W_in_temp;
    W_ext=W_out_ext+W_temp_rec;
    