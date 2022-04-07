function W_ext=samplingAll(sol,method)
%function W_ext=sampling(sol,method)

% Procedure for selecting the proper sampling procedure according to the
% chosen model

switch upper(method) 
    
    case 'UBCM' 
        
        W_ext=CM_sampling(sol);
    
    case 'UWCM'
        
        W_ext=WCM_sampling(sol);
       
    case 'DBCM'
        
        W_ext=DCM_sampling(sol);
        
    case 'DWCM'
        
        W_ext=DWCM_sampling(sol); 
       
    case 'UECM'
        
        W_ext=ECM_sampling(sol); 
       
    case 'RBCM'
        
        W_ext=RCM_sampling(sol); 
 
    case 'RWCM'
        
        W_ext=RWCM_sampling(sol); 
end

 