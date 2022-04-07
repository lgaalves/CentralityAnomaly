function  [Aout,Ain,Arec]=rec(A)
%function  [Aout,Ain,Arec]=rec(A)

n=length(A);

for i=1:n
    for j=1:n
        
        Aout(i,j)=A(i,j)*(1-A(j,i));
        Arec(i,j)=A(i,j)*A(j,i);
        
    end
end
Ain=Aout';