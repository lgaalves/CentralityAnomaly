function [sol,M,exitflag]=matrix_case(method,M)
%function [sol,M,exitflag]=matrix_case(method,M)

% Procedure for solving the constrained maximization entropy problem for
% all the models when the input data are in matrix-form

options = optimset('Display','off','Algorithm','interior-point','MaxFunEvals',(10^4),'MaxIter',5*10^4,'TolX',10^(-32),'TolFun',10^(-32),'TolCon',10^(-32));

switch upper(method) 
    
    case 'UBCM' 
        
        n = length(M);
        x0 = 0.9*ones(n,1);
        lb = zeros(n,1);
        ub=Inf*ones(n,1);
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood(x,M,method),x0,[],[],[],[],lb,ub,[],options);
    
    
    case 'UWCM' 
        
        n = length(M);
        x0 = 0.9*ones(n,1);
        lb = zeros(n,1);
        ub=ones(n,1);
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood(x,M,method),x0,[],[],[],[],lb,ub,[],options);
    
    case 'DBCM' 
        
        n = 2*length(M);
        x0 = 0.9*ones(n,1);
        lb = zeros(n,1);
        ub=Inf*ones(n,1);
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood(x,M,method),x0,[],[],[],[],lb,ub,[],options);
    
    case 'DWCM' 
        
        n = 2*length(M);
        x0 = 0.9*ones(n,1);
        lb = zeros(n,1);
        ub=[ones(n,1)];
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood(x,M,method),x0,[],[],[],[],lb,ub,[],options);  
        
    case 'UECM'
       
        n = 2*length(M);
        x0 = 0.9*ones(n,1);
        lb = zeros(n,1);
        ub=[Inf*ones(n/2,1),ones(n/2,1)];
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood(x,M,method),x0,[],[],[],[],lb,ub,[],options); 
    
    case 'RBCM'
        
        n = 3*length(M);
        x0 = 0.9*ones(n,1);
        lb = zeros(n,1);
        ub=[Inf*ones(n,1)];
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood(x,M,method),x0,[],[],[],[],lb,ub,[],options); 
        
    case 'RWCM'
        
        n = 3*length(M);
        x0 = 0.9*ones(n,1);
        lb = zeros(n,1);
        ub=[ones(n,1)];
        
        [sol,fval,exitflag] = fmincon(@(x) Likelihood(x,M,method),x0,[],[],[],[],lb,ub,[],options); 
end
        