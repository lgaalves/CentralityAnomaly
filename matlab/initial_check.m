function M=initial_check(method,Matrix)
%function M=initial_check(method,Matrix)

%Some useful checks for a correct implementation of the MAXandSAM routine

%1) Is the matrix integer?

if ~mod(sum(sum(Matrix)),1)==0
    
    M=round(Matrix);
    
    msgbox('The matrix was not integer as required from this methodology. The routine rounded it (using the command "round"). If you want to round it, please use "ctrl+c" to stop the simulation. See the "read me" file for more details.','Warning!','warn')
    
else
     M=Matrix;
end

%2) Is the matrix sysmmetric (for the undirected cases)?

if (method=='UBCM') | (method=='UWCM') | (method=='UECM')
    
    if isequal(Matrix,Matrix') == 0
      
         %msgbox('Error! The matrix should be symmetric because you are using a undirected model. See the "read me" file for more details.','ERROR!','error' )
         %error('??????????')
         
         %If you want to authomatically symmetrize it, uncomment this line:
         
         M=(M+M');    
    end
end

%3) Is the matrix symmetric but you are using an unidrected model?

if (method=='DBCM') | (method=='DWCM') | (method=='RBCM') | (method=='RWCM')
    
    if isequal(Matrix,Matrix') == 1
        
         msgbox('Error! The matrix should not be symmetric because you are using a directed model. See the "read me" file for more details.','ERROR!','error' )
         error('??????????')
    end
end