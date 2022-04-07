function Aext=RCM_sampling(sol)
%function Aext=RCM_sampling(sol)

%Procedure for drawing a matrix according to the Reciprocal Binary Configuration Model

n=length(sol);
q = n/3;
x = sol(1:q);
y = sol((q+1):2*q);
z=sol((2*q+1):end);
         
P1=zeros(q,q);
P2=zeros(q,q);
P3=zeros(q,q);

Aout=zeros(q,q);
Ain=zeros(q,q);
Arec=zeros(q,q);

for i=1:q
    for j=1:q
        if i~=j
            
            % 1) The link probabilities are computaed using the solution of
            %    the Reciprocal Binary Configuration Model
            
            d=1+x(i)*y(j)+x(j)*y(i)+z(i)*z(j);
            
            P1(i,j)=1/d;
            P2(i,j)=(x(i)*y(j))/d;
            P3(i,j)=(x(j)*y(i))/d;
           
        end
    end
end

for i=1:q
    for j=(i+1):q
        
        %2) The non-reciprocal/reciprocal (in/out) links are created using
        %   the probabilities in 1) 
        
        int=[0,P1(i,j),P1(i,j)+P2(i,j),P1(i,j)+P2(i,j)+P3(i,j),1];
        k=rand(1);
        f=is_in_class(k,int');
        
        if f==1
            Aout(i,j)=0;
            Aout(j,i)=0;
            Ain(i,j)=0;
            Ain(j,i)=0;
        else if f==2
                Aout(i,j)=1;
                Aout(j,i)=0;
                Ain(i,j)=0;
                A(j,i)=1;
            else if f==3
                    Aout(i,j)=0;
                    Aout(j,i)=1;
                    Ain(i,j)=1;
                    Ain(j,i)=0;
                else if f==4
                        Arec(i,j)=1;
                        Arec(j,i)=1;
                    end
                end
            end
        end
    end
end

Aext=Aout+Arec;
    