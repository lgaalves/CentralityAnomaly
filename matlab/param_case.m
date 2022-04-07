function [sol,fval,exitflag]=param_case(method,Par)
%function [sol,fval,exitflag]=param_case(method,Par)

% Procedure for solving the constrained maximization entropy problem for
% all the models when the input data are the constraints

options = optimset('Display','off','Algorithm','interior-point','MaxFunEvals',(10^5),'MaxIter',10^4,'TolX',10^(-32),'TolFun',10^(-32),'TolCon',10^(-32));

switch upper(method) 

    
     case 'UBCM' 
        
        n = length(Par);
        r = find(Par==0);
        
        x0 = 0.9*ones(n,1);
        x0(r)=0;
        
        lb = zeros(n,1);
        
        ub=Inf*ones(n,1);
        ub(r)=0;
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood2(x,Par,method),x0,[],[],[],[],lb,ub,[],options);
    
    
    case 'UWCM' 
        
        n =length(Par);
        r=find(Par==0);
        
        x0 = 0.9*ones(n,1);
        x0(r)=0;
        
        lb = zeros(n,1);
        
        ub=ones(n,1);
        ub(r)=0;
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood2(x,Par,method),x0,[],[],[],[],lb,ub,[],options);
    
    case 'DBCM' 
        
        n = length(Par);
        r=find(Par==0);
        
        x0 = 0.9*ones(n,1);
        x0(r)=0;
        
        lb = zeros(n,1);
        
        ub=Inf*ones(n,1);
        ub(r)=0;
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood2(x,Par,method),x0,[],[],[],[],lb,ub,[],options);
    
    case 'DWCM' 
        
        n = length(Par);
        
        x0 = 0.9*ones(n,1);
        
        lb = zeros(n,1);
        
        ub=[ones(n,1)];
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood2(x,Par,method),x0,[],[],[],[],lb,ub,[],options);  
        
    case 'UECM'
       
        n = length(Par);
      
        r=find(Par==0);
        
        x0 = 0.9*ones(n,1);
        x0(r)=0;
       
        lb = zeros(n,1);
        
        ub=[Inf*ones(n/2,1),ones(n/2,1)];
        ub(r)=0;
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood2(x,Par,method),x0,[],[],[],[],lb,ub,[],options); 
    
    case 'RBCM'
        
        n = length(Par);
        r=find(Par==0);
        
        x0 = 0.9*ones(n,1);
        x0(r)=0;
        
        lb = zeros(n,1);
        
        ub=[Inf*ones(n,1)];
        ub(r)=0;
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood2(x,Par,method),x0,[],[],[],[],lb,ub,[],options); 
        
    case 'RWCM'
        
        n = length(Par);
        
        x0 = 0.9*ones(n,1);
        
        lb = zeros(n,1);
        
        ub=[ones(n,1)];
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood2(x,Par,method),x0,[],[],[],[],lb,ub,[],options); 
        
end