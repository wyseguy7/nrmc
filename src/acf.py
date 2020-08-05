#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:07:30 2020

@author: msachs2
"""
import numpy as np
def autocorr(x, lag=100, normalized=True, xmean=None):
    """
    Computes the auto-covariance or auto-correlation of the sample trajectory stored in x up to lagtime maxlag
    Parameters:
        x : 1-dimensional numpy array of the sample trajectory
        lag: int maximum lagtime up to which the auto-correlation / auto-covariance is computed
        normalized: if True the auto-correlation is returned, otherwise the auto-covariance is returned
    """
    if isinstance(lag, int):
        lag = np.arange(0, lag+1)

    acf_vec = np.zeros(len(lag))

    if xmean is None:
        xmean = np.mean(x)

    n = x.shape[0]
    for i in range(len(lag)):
        index = np.arange(0,n-lag[i],1)
        index_shifted = np.arange(lag[i],n,1)
        acf_vec[i] = np.mean((x[index] - xmean) * (x[index_shifted] - xmean))

    if normalized:
        acf_vec/=acf_vec[0]
    return acf_vec