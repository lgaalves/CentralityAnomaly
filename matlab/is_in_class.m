function out = is_in_class(value,col_vec)
% Takes as input a value and K+1 vector x=(x_1,...,x_k+1) identifying 
% the bounds of K classes [x_j,x_j+1] and returns the class [1,...,K] which
% the value belongs to. NB: if value=x_1 then class(value)=1; if
% value=x_k+1 then class(value)=k; if value<x_1 or value>x_k+1 then
% class(value)=0;
    
    N=size(col_vec,1);
    exit_cond=0;
    myout=0;
    count=1;
    while (exit_cond==0 && count<N)
        if (value>col_vec(count) && value<=col_vec(count+1))
            myout=count;
            exit_cond=1;            
        end
        count=count+1;
    end
    
    if value==col_vec(N)
        myout=N-1;
    end
    
    out=myout;
