# Standard Library
import os
import time
import math
import subprocess
import random
from operator import itemgetter
from collections import Counter
from random import uniform

#Third Party Libraries
import networkx as nx
import igraph
import pandas as pd
import numpy as np
from scipy import stats
from scipy import optimize
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as plticker
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
# import powerlaw

from scipy import stats
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal
from scipy.stats import gaussian_kde
from matplotlib import patches
from numpy import linalg
import matplotlib.path as mpath

import  matplotlib.colors  as colors
import matplotlib.cm as cmx
import cartopy.crs as ccrs
import cartopy as cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from cartopy.io import shapereader
import geopandas


import warnings
warnings.filterwarnings('ignore') 

def stdfigsize(scale=1, nx=1, ny=1, ratio=1.3):
    """
    Returns a tuple to be used as figure size.
    -------
    returns (7*ratio*scale*nx, 7.*scale*ny)
    By default: ratio=1.3
    If ratio<0 them ratio = golden ratio
    """
    if ratio < 0:
        ratio = 1.61803398875
    return((4.66*ratio*scale*nx, 4.66*scale*ny))
    
def stdrcparams(usetex=True):
    """
    Set several mpl.rcParams and sns.set_style for my taste.
    ----
    usetex = True
    ----
    """
    sns.set_style("white")
    sns.set_style({"xtick.direction": "out",
                 "ytick.direction": "out"})
    rcparams = {
    'text.usetex': usetex,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times'],
    'axes.labelsize': 35,
    'axes.titlesize': 35,
    'legend.fontsize': 20,
    'ytick.right': 'off',
    'xtick.top': 'off',
    'ytick.left': 'on',
    'xtick.bottom': 'on',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'axes.linewidth': 2.5,
    'xtick.major.width': 1.8,
    'xtick.minor.width': 1.8,
    'xtick.major.size': 14,
    'xtick.minor.size': 7,
    'xtick.major.pad': 10,
    'xtick.minor.pad': 10,
    'ytick.major.width': 1.8,
    'ytick.minor.width': 1.8,
    'ytick.major.size': 14,
    'ytick.minor.size': 7,
    'ytick.major.pad': 10,
    'ytick.minor.pad': 10,
    'axes.labelpad': 15,
    'axes.titlepad': 15,
    'axes.spines.right': False,
    'axes.spines.top': False}

    mpl.rcParams.update(rcparams) 
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['pdf.fonttype'] = 42 

def plot_maps(Network,df,ax):
    if df is not None:
        df_names=df.copy()
    if Network=='UK':
        # get country borders
        resolution = '10m'
        category = 'cultural'
        name = 'admin_0_countries'

        shpfilename = shapereader.natural_earth(resolution, category, name)

        # read the shapefile using geopandas
        df = geopandas.read_file(shpfilename)

        # read the UK borders
        poly = df.loc[df['ADMIN'] == 'United Kingdom']['geometry'].values[0]

        ax = plt.axes(projection=ccrs.PlateCarree())


        ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='#d9d9d9', 
                          edgecolor='0.1',zorder=-1)
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([min(df_names.Lon)-3,  max(df_names.Lon)+2,min(df_names.Lat)-2, max(df_names.Lat)+2])

    if Network=='ES':
        # get country borders
        resolution = '10m'
        category = 'cultural'
        name = 'admin_0_countries'

        shpfilename = shapereader.natural_earth(resolution, category, name)

        # read the shapefile using geopandas
        df = geopandas.read_file(shpfilename)

        # read the Spain borders
        poly = df.loc[df['ADMIN'] == 'Spain']['geometry'].values[0]

        ax = plt.axes(projection=ccrs.PlateCarree())


        ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='#d9d9d9', 
                          edgecolor='0.1',zorder=-1)
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([min(df_names.Lon)-3,  max(df_names.Lon)+3,min(df_names.Lat)-2, max(df_names.Lat)+2])

    if Network=='BR':
        kww = dict(resolution='50m', category='cultural',
                  name='admin_1_states_provinces')

        states_shp = shapereader.natural_earth(**kww)
        shp = shapereader.Reader(states_shp)


        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)

        for record, state in zip(shp.records(), shp.geometries()):
            name = record.attributes['name']
            facecolor = '#d9d9d9'
            ax.add_geometries([state], ccrs.PlateCarree(),
                              facecolor=facecolor, edgecolor='black',zorder=-1)
        ax.set_extent([-74.5, -34.5, -32.5, 5.5])

    if Network=='AIR':
        ax.add_feature(cartopy.feature.LAND,alpha=0.5,zorder=-1,facecolor="#d9d9d9")
        #ax.add_feature(cartopy.feature.OCEAN,alpha=0.5,zorder=-1)
        ax.add_feature(cartopy.feature.COASTLINE,zorder=-1)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5,zorder=-1)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.95,zorder=-1)
        #ax.add_feature(cartopy.feature.RIVERS)
        # ax.set_extent([-180, 90, -90, 90])

        # shpfilename = shpreader.natural_earth(resolution='110m',
        #                                       category='cultural',
        #                                       name='admin_0_countries')
        # reader = shpreader.Reader(shpfilename)
        # paises = reader.records()


        # for country in paises:
        #         ax.add_geometries(country.geometry, ccrs.PlateCarree(),
        #                           facecolor="#d9d9d9",
        #                           label=country.attributes['ABBREV'],zorder=-1,alpha=1)
        # ax.outline_patch.set_edgecolor('white')
        return ax

def plot_network(Network,df_real,dfnames):
    if Network=='AIR':
        df_coord=labels_data(Network)

        list_of_airports=pd.read_csv("../data/airports-extended.csv",
                                     names=["id","airport.name",'city.name','country.name','IATA','ICAO','lat','long','altitude','tz.offset','DST','tz.name','airport.type','source.data'])

        list_of_airports=list_of_airports[['city.name','country.name','ICAO','lat','long']]

        list_of_airports['name']=list_of_airports['city.name'].map(str)+', '+list_of_airports['country.name']

        maplabelsAIR=dict(np.array(list_of_airports[['ICAO','name']]))
        df_coord.city=df_coord.city.map(maplabelsAIR)
        df_coord=pd.merge(df_coord,list_of_airports[['name','lat','long']],left_on='city',right_on='name',how='inner')[['id','city','lat','long']]
        df_coord.rename(columns={'lat':'Lat','long':'Lon','id':'Id'},inplace=True)


        
    # NETWORK  
    G=nx.Graph()
    G.add_edges_from(np.array(df_real[Network][['Source','Target']]))
    #positions
    nodes=list(G.nodes())
    pos={}
    
    if Network=='AIR':
        for node in nodes:
            pos[node]=list(df_coord[df_coord.Id==node+1].Lon)[0],\
                      list(df_coord[df_coord.Id==node+1].Lat)[0]

        
    else:
        for node in nodes:
            pos[node]=list(dfnames[Network][dfnames[Network].Id==node+1].Lon)[0],\
                      list(dfnames[Network][dfnames[Network].Id==node+1].Lat)[0]

    edges = list(G.edges())

    width=tuple([0.08 for item in edges])

    node_size=(np.array(list(dict(G.degree()).values())))/5
    node_size[node_size<3]=3

    cmap =plt.get_cmap('RdBu_r') 
    cNorm  = colors.Normalize(vmin=min(node_size), vmax=max(node_size))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    nodes_color=[]
    for node in node_size:
        colorVal = scalarMap.to_rgba(node)
        nodes_color.append(mpl.colors.rgb2hex(colorVal)) 

    nx.draw_networkx_nodes(G,
                    pos,
                    node_size=node_size,
                    node_color="#08306b",#nodes_color,
                    # with_labels=False,
                    #font_size=10,
                    color="k",
                    zorder=10000,
                    alpha=0.6)

    nx.draw_networkx_edges(G,
                    pos,
                    edgelist=edges,
                    edge_color="#969696", 
                    width=width, 
                    #edge_cmap=plt.cm.viridis,
                    # with_labels=False,
                    #font_size=10,
                    color="k",
                    zorder=10000,
                    alpha=0.6)

def match_datasets(data):
    allcities=pd.read_csv("../data/list_of_cities.csv",index_col=0,names=["CityUF"],encoding="utf-8")
    allcities.head()

    allcities["CityUF"]=allcities["CityUF"].str.upper()
    allcities["CityUF"]=allcities["CityUF"].str.strip()
    allcities["CityUF"]=allcities["CityUF"].str.replace(" , ",", ")
    allcities["CityUF"]=allcities["CityUF"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    setallcities=set(list(allcities.CityUF.astype(str)))

    data=data[data['ORIGEM'].isin(list(setallcities))]
    data=data[data['DESTINO'].isin(list(setallcities))]

    return data

class Data(object):
    def __init__(self):
        self.data=pd.ExcelFile('../data/brazilian_data.xls')

    def print_metrics(self):
        urbanindicators=list(self.data.sheet_names)
        return urbanindicators

    def load_metric(self,metric,state=None):
        '''
        Returns the per capita metric given an metric name
        '''
        urbanmetrics=self.data.sheet_names
        df=self.data.parse(metric)
        if state is not None:
            df=df[df.UF==state]

        return df
    
    def metadata(self,year):
        year_data=pd.DataFrame()
        for ind in self.print_metrics():
            df=self.load_metric(metric=ind)
            df=change_case(df)
            if len(set(df.columns).intersection(set([year])))==1:
                year_data[ind]=df[year]
            else:
                print("Indicator '{}' is not available for year = {}".format(ind,year))
        year_data["CityUF"]=df["CityUF"]
        return year_data
    
def change_case(df):
    df["CityUF"]=df["CityUF"].str.upper()
    df["CityUF"]=df["CityUF"].str.strip()
    df["CityUF"]=df["CityUF"].str.replace(" , ",", ")
    df["CityUF"]=df["CityUF"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    return df

def read_samples(net,number_of_samples=1):
    """
    net='AIR_agg'
    """
    sample=[]
    for i in range(1,number_of_samples+1):
        df=pd.read_csv('../samples/{}/random_{}.txt'.format(net,i),sep=' ',names=['Source','Target','Weight'],index_col=None)
        df.Source=df.Source.astype(int)-1
        df.Target=df.Target.astype(int)-1
        sample.append(df)
    return sample

def read_real_data(net):
    df=pd.read_csv('../data/network_{}.txt'.format(net),sep=' ',header=0,index_col=None)
    df.Source=df.Source.astype(int)-1
    df.Target=df.Target.astype(int)-1
    return df 

def nonweighted_ksb(df,nodes):
    g=igraph.Graph()
    g.add_vertices(nodes)
    g.add_edges(np.array(df[['Source','Target']]))
    b=g.betweenness()
    k=g.degree()
    s=g.strength()
    return k,s,b

def weighted_ksb(df,nodes):
    g=igraph.Graph(directed=False)
    g.add_vertices(nodes)
    g.add_edges(np.array(df[['Source','Target']]))
    b=g.betweenness(weights=1/np.array(df.Weight))
    k=g.degree()
    s=g.strength(weights=np.array(df.Weight))
    return k,s,b

def calculate_ksb_for_samples(df_real,sample,weighted):
    ksample=[]
    ssample=[]
    bsample=[]
    nmax=max(set(df_real.Source).union(set(df_real.Target)))
    nodes=[i for i in range(0,nmax+1)]
    for item in range(0,len(sample)):
        if weighted==False:
            k,s,b=nonweighted_ksb(sample[item],nodes=nodes)
        else:
            k,s,b=weighted_ksb(sample[item],nodes=nodes)
        ksample.append(k)
        ssample.append(s)
        bsample.append(b)
    return ksample,ssample,bsample

def calculate_and_save(net,number_of_samples):
    df_real=read_real_data('{}_weight'.format(net))

    # UBCM - non-weighted method
    ubcm_sample=read_samples('{}'.format(net),number_of_samples=number_of_samples)
    ksample,ssample,bsample=calculate_ksb_for_samples(df_real,ubcm_sample,weighted=False)
    pd.DataFrame(ksample).to_csv('../results/kbs/{}_k.csv'.format(net))
    pd.DataFrame(ssample).to_csv('../results/kbs/{}_s.csv'.format(net))
    pd.DataFrame(bsample).to_csv('../results/kbs/{}_b.csv'.format(net))

    # UECM - non-weighted method
    df=df_real.copy()
    numberofnodes=max(np.array(list(set(df_real.Source).union(set(df_real.Target)))))
    df.Source=df.Source
    df.Target=df.Target
    kw,sw,bw=weighted_ksb(df,nodes=[i for i in range(0,numberofnodes+1)])

    data=read_samples('{}_weight'.format(net),number_of_samples=number_of_samples)
    uecm_sample=[]
    for i in range(0,number_of_samples):
        df=data[i]
        df['Weight']=df['Weight']*(min(sw))
        uecm_sample.append(df)

    wksample,wssample,wbsample=calculate_ksb_for_samples(df_real,uecm_sample,weighted=True)
    pd.DataFrame(wksample).to_csv('../results/kbs/{}_weight_k.csv'.format(net))
    pd.DataFrame(wssample).to_csv('../results/kbs/{}_weight_s.csv'.format(net))
    pd.DataFrame(wbsample).to_csv('../results/kbs/{}_weight_b.csv'.format(net))

def calculate():
    for net in ['UK','ES','BR','AIR']:
        print(net)
        calculate_and_save(net,number_of_samples=10000)

def labels_data(option):
    labels=dict()
    
    dfnames=pd.read_csv('../data/nodes_label_AIR_weight.txt',sep=' ')
    dfnames.name=dfnames.name.str.upper()
    dfnames.rename(columns={'name':'city'},inplace=True)
    labels['AIR']=dfnames
    
    dfnames=pd.read_csv('../data/nodes_label_BR_Pop.txt',sep=' ')
    dfnames.Name=dfnames.Name.str.upper()
    dfnames.rename(columns={'Name':'city'},inplace=True)
    labels['BR']=dfnames
    
    dfnames=pd.read_csv('../data/nodes_label_ES_Pop.txt',sep=' ')
    dfnames.rename(columns={'Name':'city'},inplace=True)
    dfnames.city=dfnames.city.str.upper()
    dfnames.city=dfnames.city.str.replace('|',', ') 
    labels['ES']=dfnames
    
    
    dfnames=pd.read_csv('../data/nodes_label_UK_Pop.txt',sep=' ')
    dfnames.city=dfnames.city.str.upper()
    dfnames.head()
    labels['UK']=dfnames
    
    return labels[option]

def load_data_real(Network='AIR'):
    df_real=read_real_data('{}_weight'.format(Network))
    dfnames=labels_data(Network)
    
    if Network=='AIR':
        list_of_airports=pd.read_csv("../data/airports-extended.csv",
                                     names=["id","airport.name",'city.name','country.name','IATA','ICAO','lat','long','altitude','tz.offset','DST','tz.name','airport.type','source.data'])

        list_of_airports=list_of_airports[['city.name','country.name','ICAO','lat','long']]

        list_of_airports['name']=list_of_airports['city.name'].map(str)+', '+list_of_airports['country.name']

        maplabelsAIR=dict(np.array(list_of_airports[['ICAO','name']]))
        dfnames.city=dfnames.city.map(maplabelsAIR)
    if Network=='BR':
         dfnames.rename(columns={'id':"Id"},inplace=True)
    return df_real,dfnames

def ksb_load_ubcm_samples(Network):
    ksample=np.array(pd.read_csv('../results/kbs/{}_k.csv'.format(Network),index_col=0))
    ssample=np.array(pd.read_csv('../results/kbs/{}_s.csv'.format(Network),index_col=0))
    bsample=np.array(pd.read_csv('../results/kbs/{}_b.csv'.format(Network),index_col=0))
    return ksample,ssample,bsample

def ksb_load_uecm_sample(Network):
    wksample=np.array(pd.read_csv('../results/kbs/{}_weight_k.csv'.format(Network),index_col=0))
    wssample=np.array(pd.read_csv('../results/kbs/{}_weight_s.csv'.format(Network),index_col=0))
    wbsample=np.array(pd.read_csv('../results/kbs/{}_weight_b.csv'.format(Network),index_col=0))
    return wksample,wssample,wbsample

def ksb_real_data_ubcm(df_real):
    df=df_real.copy()
    numberofnodes=max(np.array(list(set(df.Source).union(set(df.Target)))))
    df.Target=df.Target
    df.Source=df.Source
    ku,su,bu=nonweighted_ksb(df,nodes=[i for i in range(0,numberofnodes+1)])
    ku=np.array(ku)
    su=np.array(su)
    bu=np.array(bu) 
    return ku,su,bu

def ksb_real_data_uecm(df_real):
    df=df_real.copy()
    numberofnodes=max(np.array(list(set(df.Source).union(set(df.Target)))))
    df.Target=df.Target
    df.Source=df.Source
    kw,sw,bw=weighted_ksb(df,nodes=[i for i in range(0,numberofnodes+1)])
    kw=np.array(kw)
    sw=np.array(sw)
    bw=np.array(bw) 
    return kw,sw,bw

def plot_ensemble_check(x,xsample,ax,samplesize=100):
    r=[]
    y=[]
    for i in range(0,samplesize):
        r.extend(x)
        y.extend(xsample[i])
    ax.plot(r,y,
            'o',
            color='#7fc97f',
            markersize=8,
            alpha=0.3,
            rasterized=True)
    ax.plot(x,np.mean(xsample,axis=0),
            's',
           color='#e7298a',
           markersize=8)

    ax.plot([i for i in range(0,int(max(x)))],[i for i in range(0,int(max(x)))],'k--',linewidth=3)
    
def plot_b_vs_k(k,b,k_sample,b_sample,ax,log=False):
    ax.plot(k,b,
            'o',
            color='#e7298a',#'#d95f02',
            markersize=8,
            alpha=0.7,
            markeredgecolor="k",
            markeredgewidth=0.4,
            rasterized=False,zorder=10)
    x=[]
    y=[]
    for i in range(0,100):
        x.extend(k_sample[i])
        y.extend(b_sample[i])
    ax.plot(x,y,'o',
            color='#7fc97f',
            markersize=8,
            alpha=0.5,
            markeredgecolor="k",
            markeredgewidth=0.4,
            rasterized=True,zorder=-1)
    # ax.plot(k,b,
    #         'o',
    #         color='#e7298a',#'#d95f02',
    #         markersize=8,
    #         alpha=0.9,
    #         markeredgecolor="k",
    #         markeredgewidth=0.7,
    #         label="Empirical data",
    #         rasterized=False)#,zorder=-1)
    
    # x=[]
    # y=[]
    # for i in range(0,100):
    #     x.extend(k_sample[i])
    #     y.extend(b_sample[i])
    # ax.plot(x,y,'o',
    #         color='#7fc97f',
    #         markersize=8,
    #         alpha=1,
    #         markeredgecolor="#7fc97f",
    #         markeredgewidth=0.7,
    #         #label="Null model",
    #         rasterized=True,zorder=-1)

    #ax.legend(fontsize=20)
    if log==True:
        ax.set_xscale('log')
        ax.set_yscale('log')
    #return ax
#     ax.set_title(Network)
#     ax.set_xlabel('Degree, $k$')
#     ax.set_ylabel('Betweenness, $b$')

def plot_b_vs_s(s,b,s_sample,b_sample,ax,log=False):
    ax.plot(s,b,
            'o',
            color='#e7298a',#'#d95f02',
            markersize=8,
            alpha=0.7,
            markeredgecolor="k",
            markeredgewidth=0.4,
            rasterized=False,zorder=10)
    x=[]
    y=[]
    for i in range(0,100):
        x.extend(s_sample[i])
        y.extend(b_sample[i])
    ax.plot(x,y,'o',
            color='#7fc97f',
            markersize=8,
            alpha=0.5,
            markeredgecolor="k",
            markeredgewidth=0.4,
            rasterized=True,zorder=-1)  
    if log==True:
        ax.set_xscale('log')
        ax.set_yscale('log') 
    
def plot_individual_cities(item,Network,k,b,k_sample,b_sample,dfnames,cmap='viridis'):
    fig = plt.figure(figsize=figsizer)
    x=np.transpose(k_sample)[item]
    y=np.transpose(b_sample)[item]
    ax = sns.kdeplot(x, y, cmap=cmap,shade=True,gridsize=100,shade_lowest=False,normalized=False)
    ax.plot(k[item],b[item],'o',color='r',markersize=12,markeredgecolor="k",markeredgewidth=1)
#     ax.set_title(dfnames.city.iloc[item]+' - '+Network)
#     ax.set_xlabel('Degree, $k$')
#     ax.set_ylabel('Betweenness, $b$')
    plt.show()
    
    
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

def plot_95fit_and_city(item,Network,k,b,k_sample,b_sample,dfnames,ax,cmap="Blues_r"):
    x=np.transpose(k_sample)[item]
    y=np.transpose(b_sample)[item] 
    # scatter the points
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    ax = sns.kdeplot(x, y, cmap=cmap,shade=True,gridsize=100,shade_lowest=False,normalized=False,rasterized=True)
    ax.scatter(x, y, c=z, s=60, edgecolor='',alpha=0.4,cmap=cmap,rasterized=True)



    (mu_r, sigma1_r,sigma2_r, alpha_r) = fit_bivariate_normal(x, y, robust=False)

    # scale=find_scale_factor(mu_r, sigma1_r,sigma2_r, alpha_r,k[item],b[item])
    scale=-2*np.log(1-0.95)
    r,t=ellipse_rotated(mu_r,scale*sigma1_r,scale*sigma2_r,alpha_r)

    plt.plot(r,t,'k-',linewidth=2)
    ax.plot(k[item],b[item],'o',color='r',
            markersize=12,markeredgecolor="k",
            markeredgewidth=1,label=dfnames.city.iloc[item].upper()+' - '+Network)


    points=np.array([[k[item],b[item]]])
    verts = np.transpose([r,t])
    path = mpath.Path(verts)
    points_inside = points[path.contains_points(points)]
    if len(points_inside) is 0:
        print('The node lies outside the 95% bounds')
    else:
        print('The node does not lie outside the 95% bounds')

#     ax.set_title(dfnames.city.iloc[item]+' - '+Network)
#     ax.set_xlabel('Degree, $k$')
#     ax.set_ylabel('Betweenness, $b$')
    
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

def bar_plot_outliers(weighted,nonweighted,ax,labels):
    x=[1,2]

    fract_w=len(weighted[weighted<1])/len(weighted)
    fract_nonw=len(nonweighted[nonweighted<1])/len(nonweighted)

    y=[fract_nonw,fract_w]
    print(y)
    ax.bar(x, 
           y,
           width=0.6,
           edgecolor='k',
           linewidth='0', 
           color=['#b2182b','#2166ac'])

    plt.xticks(x, labels, rotation='horizontal',fontsize=25)
    plt.margins(0.02)
    plt.subplots_adjust(bottom=0.15)
    ax.tick_params(direction='out', length=10, width=2, colors='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # rects = ax.patches

    # # For each bar: Place a label
    # for rect in rects:
    #     # Get X and Y placement of label from rect.
    #     y_value = rect.get_height()
    #     x_value = rect.get_x() + rect.get_width() / 2

    #     # Number of points between bar and label. Change to your liking.
    #     space = 5
    #     # Vertical alignment for positive values
    #     va = 'bottom'

    #     # If value of bar is negative: Place label below bar
    #     if y_value < 0:
    #         # Invert space to place label below
    #         space *= -1
    #         # Vertically align label at top
    #         va = 'top'

    #     # Use Y value as label and format number with one decimal place
    #     label = "{:.2f}".format(y_value)

    #     # Create annotation
    #     plt.annotate(
    #         label,                      
    #         (x_value, y_value),         
    #         xytext=(0, space),          
    #         textcoords="offset points", 
    #         ha='center',               
    #         va=va,
    #         fontsize=30) 

def plot_outliers_net(df_real,dfnames,outliers,Network):
    if Network=='AIR':
        df_coord=labels_data(Network)

        list_of_airports=pd.read_csv("../data_kaggle/airports-extended.csv",
                                     names=["id","airport.name",'city.name','country.name','IATA','ICAO','lat','long','altitude','tz.offset','DST','tz.name','airport.type','source.data'])

        list_of_airports=list_of_airports[['city.name','country.name','ICAO','lat','long']]

        list_of_airports['name']=list_of_airports['city.name'].map(str)+', '+list_of_airports['country.name']

        maplabelsAIR=dict(np.array(list_of_airports[['ICAO','name']]))
        df_coord.city=df_coord.city.map(maplabelsAIR)
        df_coord=pd.merge(df_coord,list_of_airports[['name','lat','long']],left_on='city',right_on='name',how='inner')[['id','city','lat','long']]
        df_coord.rename(columns={'lat':'Lat','long':'Lon','id':'Id'},inplace=True)

    G=nx.Graph()
    G.add_edges_from(np.array(df_real[Network][['Source','Target']]))
    nodes=list(G.nodes())
    pos={}
    if Network=='AIR':
        for node in nodes:
            pos[node]=list(df_coord[df_coord.Id==node+1].Lon)[0],\
                  list(df_coord[df_coord.Id==node+1].Lat)[0]

    else:
        for node in nodes:
            pos[node]=list(dfnames[Network][dfnames[Network].Id==node+1].Lon)[0],\
                  list(dfnames[Network][dfnames[Network].Id==node+1].Lat)[0]

    node_size=(np.array(list(dict(G.degree()).values())))/5
    node_size_all=node_size.copy()
    node_size_all[node_size_all<3]=3
    
    nx.draw_networkx_nodes(G,
                pos,
                node_size=node_size_all,
                node_color="#08306b",#nodes_color,
                # with_labels=False,
                #font_size=10,
                color="k",
                zorder=10000,
                alpha=0.6)
    node_size[node_size<100]=20

    k=G.subgraph(np.array(nodes)[outliers[Network]<1])
    node_size=np.array(node_size)[outliers[Network]<1]
    nx.draw_networkx_nodes(k,
                    pos,
                    node_size=node_size,
                    node_color="r",#nodes_color,
                    # with_labels=False,
                    #font_size=10,
                    color="k",
                    zorder=10000,
                    alpha=0.6)

def plot_network_selected_nodes(Network,df_real,list_of_nodes,dfnames,color):
    if Network=='AIR':
        df_coord=labels_data(Network)

        list_of_airports=pd.read_csv("../data_kaggle/airports-extended.csv",
                                     names=["id","airport.name",'city.name','country.name','IATA','ICAO','lat','long','altitude','tz.offset','DST','tz.name','airport.type','source.data'])

        list_of_airports=list_of_airports[['city.name','country.name','ICAO','lat','long']]

        list_of_airports['name']=list_of_airports['city.name'].map(str)+', '+list_of_airports['country.name']

        maplabelsAIR=dict(np.array(list_of_airports[['ICAO','name']]))
        df_coord.city=df_coord.city.map(maplabelsAIR)
        df_coord=pd.merge(df_coord,list_of_airports[['name','lat','long']],left_on='city',right_on='name',how='inner')[['id','city','lat','long']]
        df_coord.rename(columns={'lat':'Lat','long':'Lon','id':'Id'},inplace=True)


        
    # NETWORK  
    G=nx.Graph()
    G.add_edges_from(np.array(df_real[Network][['Source','Target']]))
    #positions
    nodes=list(G.nodes())
    pos={}
    
    if Network=='AIR':
        for node in nodes:
            pos[node]=list(df_coord[df_coord.Id==node+1].Lon)[0],\
                      list(df_coord[df_coord.Id==node+1].Lat)[0]

    else:
        for node in nodes:
            pos[node]=list(dfnames[Network][dfnames[Network].Id==node+1].Lon)[0],\
                      list(dfnames[Network][dfnames[Network].Id==node+1].Lat)[0]

    edges = list(G.edges())

    width=tuple([0.08 for item in edges])

    node_size=(np.array(list(dict(G.degree()).values())))/5
    node_size[node_size<10000]=100

    contained = [item in list_of_nodes for item in nodes]

    k=G.subgraph(np.array(nodes)[contained])
    node_size=np.array(node_size)[contained]
    nx.draw_networkx_nodes(k,
                    pos,
                    node_size=node_size,
                    node_color=color,#nodes_color,
                    # with_labels=False,
                    #font_size=10,
                    color="k",
                    zorder=10000,
                    alpha=0.8
                    )
