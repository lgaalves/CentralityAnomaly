  function output = MAXandSAM(method,Matrix,Par,List,eps,sam,x0new)


% Routine for solving max-entropy models with different constraints and for
% different kinds of input-data.
% 
% 1)"method" is a string indicating the acronym for models according to:
%
% 'UBCM' - Undirected Binary Configuration Model
%         
% 'UWCM' - Undirected Weighted Configuration Model
%          
% 'DBCM' - Directed Binary Configuration Model
%         
% 'DWCM'-  Directed Weighted Configuration Model
%        
% 'UECM'-  Undirected Enhanced Configuration Model
%          
% 'RBCM'-  Reciprocal Binary Configuration Model
%         
% 'RWCM'-  Reciprocal Weighted Configuration Model
%        
%
% 2) There are three possibilities for input-data:
%  
%   1) "Matrix" --> Binary or Weighted Matrix according to the chosen model
%                 (user's choice) 
%   2) "List"   --> Edge List
%   3) "Par"    --> Constraint sequences (degree, strength...)
%                 !!NB!!: it must be a unique column vector.
%                 For directed cases the order is [cons_out;cons_in]
%                 For reciprocal model the order is
%                 [cons_in;cons_out;cons_rec];
%  
% NB: only one choice is possible, the other two should be replaced by [].
%
%    
% 3)"eps" is a control parameter for the maximum relative error between the
% observed and the expected value of the constraint(s).
%  
% 4) "sam" is a boolean parameter allowing the user to choose to extract a
% certain number of matrices from the chosen ensamble. "0" corresponds to no
% sampling, "1" to sampling. The algortihm will ask the user to introduce the 
% number of matrics to draw from the specific distribution.
%
% 5) "x0new" is an optional parameter introduced for a very specific case.
%
% Please, see the "Read_me" file for more details.
%
% The MAX&SAM method was introduced in [1] and the ensembles it
% generates were defined in [2] (BCM, WCM, RCM), [3] (WRCM), [4] (ECM).
% You can use this routine freely, provided that in all your
% publications and presentations showing any result that builds upon the
% use of this routine you ALWAYS cite ref. [1] or its published version
% (as soon as it is available) and you ALWAYS cite the relevant
% references (either [2], [3], [4], or combinations of them) defining
% the specific ensemble(s) you select in the routine.
% 
% [1] Tiziano Squartini, Rossana Mastrandrea, Diego Garlaschelli,
% "Unbiased sampling of network ensembles",
% http://arxiv.org/abs/1406.1197
% (Please replace this preprint version with the final published
% version, as soon as the latter is available.)
% 
% [2] Tiziano Squartini, Diego Garlaschelli, "Analytical
% maximum-likelihood method to detect patterns in real networks", New
% Journal of Physics 13: 083001, (2011).
% 
% [3] Tiziano Squartini, Francesco Picciolo, Franco Ruzzenenti, Diego
% Garlaschelli, "Reciprocity of weighted networks", Scientific Reports
% 3: 2729 (2013).
% 
% [4] Rossana Mastrandrea, Tiziano Squartini, Giorgio Fagiolo, Diego
% Garlaschelli, "Enhanced reconstruction of weighted networks from 
% strengths and degrees" New Journal of Physics 16: 043022, (2014).
%
% Copyright, Mastrandrea Rossana, 2014 (rossmastrandrea@gmail.com) 




%msgbox('You can use this routine freely, provided that in all your publications and presentations showing any result that builds upon the use of this routine you ALWAYS cite references ([1] and either [2], [3], [4], or combinations of them) defining the specific ensemble(s) you select in the routine. You find them in the commented part or writing "help MAXandSAM" in the command window.')



tic;

%This part is designed for some specific cases that show convergence
%problem. For more details see the "Read_me" file

if nargin==6

% Input data in matrix-form

if  isempty(Par) && isempty(List)
    
    M=initial_check(method,Matrix);
    
    [sol fval exitflag]=matrix_case(method,M);
    
    arg=1;
    
    err = system_check(method,sol,M,arg);

end

% Input data in edge-list form

if isempty(Matrix) && isempty(Par)
    Matrix=edges2matrix(List);
    
    M=initial_check(method,Matrix);
    
    [sol fval exitflag]=matrix_case(method,M);
    
    arg=1;
    
    err = system_check(method,sol,M,arg);

end

%Input data as contraints

if isempty(Matrix) && isempty(List)
    
    M=Par;
    
    [sol,fval,exitflag]=param_case(method,M);
    
    arg=1;
    
    err = system_check(method,sol,M,arg);
    
end

%Not solvable cases

if exitflag == -2
    error('No feasible point was found')
end

%Check for choosing the second step (if any): solution of the related
%system
            
if err < eps
          
    str = sprintf('MAXIMUM FOUND. Constraints preserved with maximum relative error: %d', err);
    disp(str);
    output=sol;

else
    
    %step 1) for avoiding to do many iterations when the error goes to 0
    %quickly
    
   
    x0=sol;
    options=optimset('Display','off','Algorithm','trust-region-reflective','MaxFunEvals',(10^4),'MaxIter',10^4,'TolX',10^(-32),'TolFun',10^(-32),'TolCon',10^(-32));
    
    [solnew,fval,exitflag] = fsolve(@(x) system_check(method,x,M,0),x0,options);
    
    err2 = system_check(method,solnew,M,1);
    
    % step 2) if the error is still not small the solution is used as input
    % for continuing to solve the srelates system
    
    if err2 < eps
        
        str = sprintf('SYSTEM SOLVED. Constraints preserved with maximum relative error: %d', err2);
        disp(str);
        output=solnew;
    
    else
        
        x0=solnew;
        options=optimset('Display','off','Algorithm','trust-region-reflective','MaxFunEvals',(10^5),'MaxIter',10^5,'TolX',10^(-32),'TolFun',10^(-32),'TolCon',10^(-32));
            
        [solnew2,fval,exitflag] = fsolve(@(x) system_check(method,x,M,0),x0,options);
        err3 = system_check(method,solnew2,M,1);
        
        str = sprintf('SYSTEM SOLVED. Constraints preserved with maximum relative error: %d', err3);
        disp(str);
        output=solnew2;
    end
end

end

    
    

toc;

if nargin==7
    x0=x0new; 
    M=Par;
    options=optimset('Display','off','Algorithm','trust-region-reflective','MaxFunEvals',(10^4),'MaxIter',10^5,'TolX',10^(-32),'TolFun',10^(-32),'TolCon',10^(-32));
    [solnew3,fval,exitflag] = fsolve(@(x) system_check(method,x,M,0),x0,options);
    err4 = system_check(method,solnew3,M,1);
    str = sprintf('SYSTEM SOLVED. Constraints preserved with maximum relative error: %d', err4);
    disp(str);
    output=solnew3;
end

if sam==1
    %reply=input('Do you want to draw a matrix according to the chosen model? Y/N \n','s');

    %if reply=='Y' | reply=='y'
    
    num=input('How many matrices do you want to draw? \n');
    for i=1:num
        W_ext=samplingAll(output,method);
        name=strcat('sample-',num2str(i));
        dlmwrite(strcat(name,'.txt'),W_ext)
        Sampling{1,i}=W_ext;
    end
    
    %The extracted matrices are authomatically saved in "Sampling.mat"
    
%     savefile='Sampling.mat';
%     save(savefile,'Sampling') 
end
    
        
        
        