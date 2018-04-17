#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:32:50 2018

@author: dannem
"""
def foldMatDN(data,els='all',stim='all',domain='all',blocks='all',ilust=False):
    # channel * freq * id * block
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    if 'all' in str(els):
        els=np.arange(0,data.shape[0])
    if 'all' in str(stim):
        stim=np.arange(0,data.shape[2])
    if 'all' in str(domain):
        domain=np.arange(0,data.shape[1])
    if 'all' in str(blocks):
        blocks=np.arange(0,data.shape[3])
    data=data[np.ix_(els,domain,stim,blocks)]
    labels=np.tile(np.arange(1,data.shape[2]+1),data.shape[3])
    data=data.reshape((data.shape[0]*data.shape[1],data.shape[2]*data.shape[3]),order='F')
    data=data.transpose()
    if ilust==True:
        fig = plt.gcf()
        fig.set_size_inches(20, 15)
        sns.heatmap(data,vmin=0, vmax=0.74)
        plt.show()
    return (data,labels)


def loadDataDN(fileName, folder='default',printSize=True):
    """Written by Nemrodov Dan
    Imports data from the lwdata.mat structure
    Example: data=af.loadDataDN('S09_fft.mat')
    """
    import platform
    import scipy.io
    import numpy as np
    if 'default' in folder:
        if 'nestor' in platform.uname()[1]:
            folder='/Users/dannem/Documents/DataAnalysis'
        elif 'Dell_DN' in platform.uname()[1]:
            folder='C:/Users/Dan/Documents/MATLAB'
    if '.mat' in fileName:
        fileName=fileName[:-4]
    mat=scipy.io.loadmat(folder+'/'+fileName+'.mat')
    if printSize:
         print(type(mat))
         print(mat.keys())
         print(np.shape(mat[fileName]))
    try:
         data = mat[fileName]
    except:
         data=mat
    del mat
    return data