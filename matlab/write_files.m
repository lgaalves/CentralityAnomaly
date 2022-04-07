function write_files(outputs,method,net_label,num)
switch upper(method) 
    case 'UBCM' 
        path=strcat('../spatial_model/UBCM/',net_label);
    case 'UECM'
        path=strcat('../spatial_model/UECM/',net_label);
end

path=strcat(path,'-')

for i=1:num
    W_ext=samplingAll(outputs,method);
    edges=adj2edge(W_ext);
    name=strcat(path,num2str(i));
    dlmwrite(strcat(name,'.txt'),edges)
end
end