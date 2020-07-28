#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:07:30 2020

@author: msachs2
"""
import numpy as np
def autocorr(x, maxlag=100, normalized=True):
    """
    Computes the auto-covariance or auto-correlation of the sample trajectory stored in x up to lagtime maxlag
    Parameters:
        x : 1-dimensional numpy array of the sample trajectory
        maxlag: int maximum lagtime up to which the auto-correlation / auto-covariance is computed
        normalized: if True the auto-correlation is returned, otherwise the auto-covariance is returned
    """
    acf_vec = np.zeros(maxlag)
    xmean = np.mean(x)
    n = x.shape[0]
    for lag in range(maxlag):
        index = np.arange(0,n-lag,1)
        index_shifted = np.arange(lag,n,1)
        acf_vec[lag] = np.mean((x[index ]-xmean)*(x[index_shifted]-xmean))
    
    if normalized:
        acf_vec/=acf_vec[0]
    return acf_vec 