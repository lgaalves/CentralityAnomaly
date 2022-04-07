function SaveSpatialModelSamples(net_label,num)    
pairks=load(strcat('../spatial_model/',strcat(net_label,'.txt')))
outputs = MAXandSAM('UECM',[],pairks,[],10^(-6),0);
length(outputs)
write_files(outputs,'UECM',net_label,num);
l=length(pairks);
outputs = MAXandSAM('UBCM',[],pairks(1:l/2),[],10^(-6),0);
length(outputs)
write_files(outputs,'UBCM',net_label,num)