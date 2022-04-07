function [Wout,Win,Wrec]=recW(W)
%function [Wout,Win,Wrec]=recW(W)

n=length(W);

for i=1:n
    for j=1:n
        if i ~= j
            
            Wrec(i,j)=min(W(i,j), W(j,i));
            Wout(i,j)=W(i,j)-Wrec(i,j);
            
        end
    end
end

Win=Wout';