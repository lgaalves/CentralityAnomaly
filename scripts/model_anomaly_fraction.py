#!/usr/bin/python

#Third Party Libraries
import pandas as pd
import numpy as np
from scipy import stats
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal
from scipy.stats import gaussian_kde
import matplotlib.path as mpath

def net_name(init_nodes,final_nodes,m,delta,w0,rc):
    file='{}_{}_{}_{}_{}_{}'.format(init_nodes,final_nodes,m,round(delta,2),w0,round(rc,4))
    return file

def read_ksb_samples(method,file):
    ksample=np.array(pd.read_csv('../spatial_model/{}/k_{}.csv'.format(method,file),index_col=0))
    ssample=np.array(pd.read_csv('../spatial_model/{}/s_{}.csv'.format(method,file),index_col=0))
    bsample=np.array(pd.read_csv('../spatial_model/{}/b_{}.csv'.format(method,file),index_col=0))
    return ksample,ssample,bsample

def read_ksb_model(method,file):
    k,s,b=np.array(pd.read_csv('../spatial_model/{}_ksb_{}.txt'.format(method,file),index_col=0).T)
    return k,s,b

    
def ellipse_rotated(mu_r,sigma1_r,sigma2_r,alpha_r):
    xcenter, ycenter = mu_r
    width, height =sigma1_r,sigma2_r
    angle = alpha_r*180./np.pi

    theta = np.arange(0.0, 360.0, 1.0)*np.pi/180.0
    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)

    rtheta = np.radians(angle)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
        ])

    x, y = np.dot(R, np.array([x, y]))
    x += xcenter
    y += ycenter
    return x,y

def find_nodes_outside_the_p_bound(k,b,k_sample,b_sample,p=0.95):
    outliers=[]
    for item in range(0,len(k)):
        x=np.transpose(k_sample)[item]
        y=np.transpose(b_sample)[item]
        (mu_r, sigma1_r,sigma2_r, alpha_r) = fit_bivariate_normal(x, y, robust=False)
        scale=-2*np.log(1-p)
        r,t=ellipse_rotated(mu_r,scale*sigma1_r,scale*sigma2_r,alpha_r)

        points=np.array([[k[item],b[item]]])
        verts = np.transpose([r,t])
        path = mpath.Path(verts)
        points_inside = points[path.contains_points(points)]
        outliers.append(len(points_inside))
    outliers=np.array(outliers)
    return outliers

def main():
    init_nodes = 5
    final_nodes = 1000
    m = 4
    w0=1
    n_samples=10000
    nodes=[i for i in range(1,final_nodes)]
    
    results_ubcm=[]
    results_uecm=[]
    for rc in [0.01,1,10]:
        for delta in [0.01,1,10]:
            file=net_name(init_nodes,final_nodes,m,delta,w0,rc)
            print(file)
            k,s,b=read_ksb_model('UBCM',file)
            ksample,ssample,bsample=read_ksb_samples('UBCM',file)
            out=find_nodes_outside_the_p_bound(k,b,ksample,bsample,p=0.95)
            fraction=len(out[out<1])/len(out)
            results_ubcm.append([delta,rc,fraction])

            k,s,b=read_ksb_model('UECM',file)
            ksample,ssample,bsample=read_ksb_samples('UECM',file)
            out=find_nodes_outside_the_p_bound(k,b,ksample,bsample,p=0.95)
            fraction=len(out[out<1])/len(out)
            results_uecm.append([delta,rc,fraction])
    results_ubcm=pd.DataFrame(results_ubcm,columns=['delta','rc','fraction'])
    results_ubcm.to_csv('../spatial_model/UBCM/results_ubcm.csv')

    results_uecm=pd.DataFrame(results_uecm,columns=['delta','rc','fraction'])
    results_uecm.to_csv('../spatial_model/UECM/results_uecm.csv')


if __name__ == "__main__":
    main()
