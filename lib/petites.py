#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:30:34 2017

@author: parag rastogi
"""

import random
from copy import deepcopy
import numpy as np
import math
import pandas as pd

# Declare a function for smoothing data, to be used later.

def smoother(xin, lags):
    
    xout = np.zeros_like(xin)
    
    # Convert even numbers to odd.
    if np.mod(lags, 2) == 0:
        lags = lags + 1
        
    halfspan = int(np.floor(lags/2))
        
    for n, x in enumerate(xin):
        
        if (n < halfspan):
            xout[n] = np.nanmean(xin[0:halfspan])
        
        elif (n > (xin.shape[0] - halfspan)):
            xout[n] = np.nanmean(xin[-halfspan:-1])
        
        else:
            lb = int(n - halfspan)
            ub = int(n + halfspan)
            xout[n] = np.nanmean(xin[lb:ub])
        
        # End IF statement.
        
    # End FOR loop.
    
    return xout

# Define a function that calculates a "circular" rolling mean. That is, it pads 
# the first day of the year with the last day to ensure that the beginning is smooth.

def circ_rolling_mean(df, window=730):
    ''' Circular moving average filter window.'''

    window_copy = deepcopy(df.iloc[-window:])
    
    if isinstance(window_copy.index, pd.core.index.MultiIndex): 
    
        first_index = set(window_copy.index.get_level_values(0))
        
        w_index = window_copy.index.get_level_values(1) - pd.DateOffset(hours=8760)
        
    else:
        w_index = window_copy.index - pd.DateOffset(hours=8760)
    
    window_copy.index = w_index
        
    df_copy = (pd.concat([window_copy, df])).rolling(window=window).mean()
    
    return df_copy[window:]


def ecdf(x, bins=25):
    hist, bin_edges = np.histogram(x, bins=bins)

    hist = np.cumsum(hist/sum(hist))/(bin_edges[1]-bin_edges[0])

    return hist, bin_edges


def epdf(x, bins=25):
    hist, bin_edges = np.histogram(x, bins=bins)

    hist = hist/sum(hist)/(bin_edges[1]-bin_edges[0])

    return hist, bin_edges


def rel_hist(x, bins=25):
    hist, bin_edges = np.histogram(x, bins=bins)

    hist = hist/sum(hist)

    return hist, bin_edges


def remove_leap_day(df):
    '''Removes leap day using time index.'''

    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        df_return = df[~((df.index.month == 2) & (df.index.day == 29))]
    elif isinstance(df, pd.DatetimeIndex):
        df_return = df[~((df.month == 2) & (df.day == 29))]

    return df_return

# ----------- END remove_leap_day function. -----------


def euclidean(x, y):

    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

# ----------- END euclidean function. -----------


def dd(xin, bp, freq='H'):

    if isinstance(bp, list):
        hdd = (bp[1] - xin)
        cdd = (xin - bp[0])
        hdd=hdd[hdd>0].sum()
        cdd=cdd[cdd>0].sum()
    else:
        dd = xin - bp
        hdd = np.abs(dd[dd<0].sum())
        cdd = dd[dd>0].sum()
        
    # Adjust for frequency of incoming data.
    if freq == 'H':
        hdd = hdd/24
        cdd = cdd/24

    return cdd, hdd

