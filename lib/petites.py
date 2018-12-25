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
# from scipy import interpolate
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from datetime import timedelta

# Constants for Eq. 5, Temperature -200°C to 0°C.
FROZEN_CONST = [-5.6745359 * 10**3, 6.3925247, -9.6778430 * 10**-3,
                6.2215701 * 10**-7, 2.0747825 * 10**-9,
                -9.4840240 * 10**-13, 4.1635019]

# Constants for Eq. 6, Temperature 0°C to 200°C.
LIQUID_CONST = [-5.8002206 * 10**3, 1.3914993, -4.8640239 * 10**-2,
                4.1764768 * 10**-5, -1.4452093 * 10**-8, 6.5459673]


def check_pressure_unit(x, unitin):
    '''This function checks the units of the incoming pressure time series by checking the number of digits before the decimal point in the mean. Then, comparing to the standard pressure in Pa (101325 Pa = 6 digits), we can determine the conversion factor (power of exponent).'''
    
    if unitin == 'mbar':
        y = x*100
    elif unitin == 'kPa':
        y = x*1000
            
    return y


def ecdf(x, bins=25):
    hist, bin_edges = np.histogram(x, bins=bins)

    hist = np.cumsum(hist/sum(hist))/(bin_edges[1]-bin_edges[0])

    return hist, bin_edges


def epdf(x, bins=25):
    hist, bin_edges = np.histogram(x, bins=bins)

    hist = hist/sum(hist)/(bin_edges[1]-bin_edges[0])

    return hist, bin_edges


def setseed(randseed):
    '''Seed random number generators. Called as a function in main indra
    script once and only once.'''

    np.random.seed(randseed)
    random.seed = randseed

# ----------- END setseed function. -----------


def percentilecleaner(datain, xy_train, bounds=None, interp_method='linear'):
    '''Generic cleaner based on percentiles. Needs a time series / dataset
       and cut-off percentiles. Also needs the name of the variable (var) in
       the incoming dataframe. This function will censor the data outside
       those percentiles and interpolate the missing values using linear
       interpolation.'''

    if bounds is None:
        bounds = [1, 99]

    dataout = deepcopy(datain)

    rec_quantiles = xy_train.quantile([x/100 for x in bounds])

    dataout = dataout.mask(np.logical_or(dataout < rec_quantiles.iloc[0], dataout > rec_quantiles.iloc[1]))

    # Interpolate.
    dataout = dataout.interpolate(method=interp_method).fillna(method='bfill').fillna(method='ffill')

    # Put the text and date columns back in.
    text_colnames = dataout.select_dtypes(include=object, exclude=np.number).columns
    date_colnames = ('year', 'month', 'day', 'hour')
    dataout.loc[:, text_colnames] = datain.loc[:, text_colnames]
    dataout.loc[:, date_colnames] = datain.loc[:, date_colnames]

    return dataout

# ----------- END percentilecleaner function. -----------
    

def percentileCleanerMonthly(datain, xy_train, bounds=None, interp_method='linear', varstochange=('tdb', 'tdp', 'rh', 'atmpr', 'wspd', 'wdir')):
    '''Generic cleaner based on percentiles. Needs a time series / dataset
       and cut-off percentiles. Also needs the name of the variable (var) in
       the incoming dataframe. This function will censor the data outside
       those percentiles and interpolate the missing values using linear
       interpolation.'''

    if bounds is None:
        bounds = [1, 99]

    dataout = deepcopy(datain)
    
    varsindata = [x for x in varstochange if x in dataout.columns]
        
    for month in range(1,13):
        
        data_this_month = dataout.loc[dataout.index.month==month, varsindata]

        rec_quantiles = xy_train.loc[xy_train.index.month==month, varsindata].quantile([x/100 for x in bounds])
    
        data_this_month = data_this_month.mask(np.logical_or(data_this_month < rec_quantiles.iloc[0], data_this_month > rec_quantiles.iloc[1]))
    
        # Interpolate.
        dataout.loc[dataout.index.month==month, varsindata] = data_this_month.interpolate(method=interp_method).fillna(method='bfill').fillna(method='ffill')
    
    # Put the text and date columns back in.
    text_colnames = dataout.select_dtypes(include=object, exclude=np.number).columns
    date_colnames = ('year', 'month', 'day', 'hour')
    dataout.loc[:, text_colnames] = datain.loc[:, text_colnames]
    dataout.loc[:, date_colnames] = datain.loc[:, date_colnames]

    return dataout

# ----------- END percentilecleaner function. -----------


def firstdiffcleaner(datain, xy_train, bounds = None, placeholder_cols = ['year', 'month', 'day', 'hour', 'minute'],
                     use_cols = None, smoothing_window_hrs = 6, fit_window_hrs = 24, poly_order = 3):
    '''Generic cleaner based on first difference. Needs a time series / dataset
       and cut-off percentiles. If the names of the affected variables (var)
       are not provided, then all variables in datain will be censored.
       This function will censor the data outside the given percentiles of
       the first difference and interpolate the missing values using linear
       interpolation.'''

    if bounds is None:
        bounds = [1, 99]

    dataout = deepcopy(datain)
    # dataout = dataout.loc[~((dataout.index.month == 2) & (dataout.index.day == 29)), :]

    # Ensure that columns are actually in data frame
    placeholder_cols_checked = [c for c in placeholder_cols if c in dataout.columns]

    # Remove non-numeric columns as they can't be diff'ed
    non_numeric_cols_syn = dataout.select_dtypes(include = None, exclude = np.number)
    placeholder_cols_syn = dataout.loc[:, placeholder_cols_checked]
    dataout = dataout.select_dtypes(include = np.number, exclude = None)
    dataout.drop(placeholder_cols_checked, axis = 1, inplace = True)
    numeric_cols_rec = xy_train.select_dtypes(include = np.number, exclude = None)

    diff_syn = dataout.diff(periods = 1, axis = 0)
    # First values comes out as NaN - set to zero.
    diff_syn.iloc[0, :] = 0

    # Keep only the columns from xy_train that exist in datain/dataout.
    if len(dataout.columns) < len(xy_train.columns):
        numeric_cols_rec.drop([x for x in numeric_cols_rec.columns if x not in dataout.columns], axis = 1)

    diff_rec = numeric_cols_rec.diff(periods = 1, axis = 0)
    diff_rec.iloc[0, :] = diff_rec.iloc[1, :]
    diff_rec_quantiles = diff_rec.abs().quantile(max(bounds) / 100)

    dataout = dataout.mask(diff_syn.abs() > diff_rec_quantiles, other = np.nan)

    # Set up windows
    smoothing_window = timedelta(hours = smoothing_window_hrs / 2)
    fit_window = timedelta(hours = fit_window_hrs / 2)

    # Calculate time since start to use as x values of fit
    dataout['total_secs'] = (dataout.index - dataout.index[0]).total_seconds()

    # Eligible columns
    eligible_use_cols = [c for c in dataout.columns if c in use_cols]

    for column_name in eligible_use_cols:
        dataout_column = deepcopy(dataout.loc[:, [column_name, 'total_secs']])

        #print('I have found a maximum of {} consecutive NaN values for column {}'.format(max(dataout_column[column_name].isnull().astype(int).groupby(dataout_column[column_name].notnull().astype(int).cumsum()).sum()), column_name))

        for nan_index in dataout_column[np.isnan(dataout_column[column_name])].index:
            # Set up the fit windows etc.
            smoothing_window_left = dataout.index[0]
            smoothing_window_right = dataout.index[0]
            fit_window_left = dataout.index[0]
            fit_window_right = dataout.index[0]

            if nan_index > smoothing_window_right:
                # Calculate windows
                smoothing_window_left = nan_index - smoothing_window
                smoothing_window_right = nan_index + smoothing_window
                fit_window_left = nan_index - fit_window
                fit_window_right = nan_index + fit_window

                if poly_order >= 0:
                    model = Pipeline([('poly', PolynomialFeatures(degree = poly_order)),
                                      ('regressor', LinearRegression())])

                    model.fit(X = np.reshape(dataout_column.loc[~np.isnan(dataout_column[column_name]), 'total_secs'][fit_window_left:fit_window_right].values, (-1, 1)),
                              y = dataout_column.loc[~np.isnan(dataout_column[column_name]), column_name][fit_window_left:fit_window_right])

                    df_model = pd.DataFrame({'y_new': model.predict(np.reshape(dataout_column.loc[fit_window_left:fit_window_right, 'total_secs'].values, (-1, 1))),
                                             'y_old': dataout_column.loc[~np.isnan(dataout_column[column_name]), column_name][fit_window_left:fit_window_right]},
                                             index = dataout_column.loc[fit_window_left:fit_window_right, :].index)

                    # Calculate triangular distribution weights based on NaN values
                    df_model['y_new_weight'] = 0
                    df_model.loc[smoothing_window_left:smoothing_window_right, 'y_new_weight'] = np.arange(start = 0, stop = df_model[smoothing_window_left:smoothing_window_right].shape[0], step = 1)
                    df_model.loc[smoothing_window_left:smoothing_window_right, 'y_new_weight'] = np.where(df_model[smoothing_window_left:smoothing_window_right].index <= nan_index,
             2 * df_model.loc[smoothing_window_left:smoothing_window_right, 'y_new_weight'] / (df_model[smoothing_window_left:smoothing_window_right].shape[0] - 1),
             2 - 2 * df_model.loc[smoothing_window_left:smoothing_window_right, 'y_new_weight'] / (df_model[smoothing_window_left:smoothing_window_right].shape[0] - 1))

                    # Set any old values which were NaN to zero
                    df_model.loc[np.isnan(df_model['y_old']), 'y_old'] = 0

                    # Now calculate the new value of y using the weights
                    df_model['y_weighted'] = df_model['y_old']
                    df_model.loc[smoothing_window_left:smoothing_window_right, 'y_weighted'] = (df_model.loc[smoothing_window_left:smoothing_window_right, 'y_new'] * df_model.loc[smoothing_window_left:smoothing_window_right, 'y_new_weight']) + (df_model.loc[smoothing_window_left:smoothing_window_right, 'y_old'] * (1 - df_model.loc[smoothing_window_left:smoothing_window_right, 'y_new_weight']))

                    # Write back to original array
                    dataout_column.loc[smoothing_window_left:smoothing_window_right, column_name] = df_model['y_weighted']

                elif poly_order < 0:
                    # Moving average.

                    if (dataout.loc[smoothing_window_left:smoothing_window_right]).shape[0] < smoothing_window_hrs or (dataout.loc[fit_window_left:fit_window_right]).shape[0] < fit_window_hrs:
                        find_nan_index = np.where(dataout.index==nan_index)[0][0]
                        nan_range_fit = np.arange(max(0, find_nan_index - math.floor(fit_window_hrs/2)), min(dataout_column.shape[0], find_nan_index + math.floor(fit_window_hrs/2)))
                        nan_range_smooth = np.arange(max(0, find_nan_index - math.floor(smoothing_window_hrs/2)), min(dataout_column.shape[0], find_nan_index + math.floor(smoothing_window_hrs/2)))

                        df_model = dataout_column.iloc[nan_range_fit, :].rolling(window=smoothing_window_hrs, win_type='triang', min_periods=1).mean()

                        nan_range_df = range(max(0, np.where(df_model.index==nan_index)[0][0] - math.floor(smoothing_window_hrs/2)), min(df_model.shape[0], np.where(df_model.index==nan_index)[0][0] + math.floor(smoothing_window_hrs/2)))

                        dataout_column.iloc[nan_range_smooth, :] = df_model.iloc[nan_range_df, :]

                    else:
                        df_model = dataout_column.loc[fit_window_left:fit_window_right, column_name].rolling(window=smoothing_window_hrs, win_type='triang', min_periods=1, on='total_sces').mean()
                        dataout_column.loc[smoothing_window_left:smoothing_window_right, column_name] = df_model.loc[smoothing_window_left:smoothing_window_right]

                #print('Column {}, nan_index = {}'.format(column_name, nan_index))

        dataout[column_name] = dataout_column

        """plt.plot(dataout_column.loc[~np.isnan(dataout_column[column_name]), column_name][fit_window_left:fit_window_right], color = 'grey', marker = 'o', linestyle = 'None')
        plt.plot(df_model['y_new'], color = 'b', marker = 'None', linestyle = 'dashed')
        plt.plot(df_model['y_weighted'], color = 'red', marker = 'o', linestyle = 'None')
        plt.xticks(rotation=90)
        """

        """
        start = '2223-01-30'
        end = '2223-02-03'
        var = 'rh'

        plt.plot(datain.loc[start:end, var], color = 'r', marker = 'o', linestyle = 'None', markersize = 2)
        plt.plot(dataout.loc[start:end, var], color = 'b', marker = 'o', linestyle = 'None', markersize = 2)
        """

    # There are still occasional nans here for some reason. Should look into this eventually but for now I am interpolating linearly. (PARAG)
    dataout = dataout.interpolate(method = 'linear').fillna(method = 'bfill').fillna(method = 'ffill')

    # Put the text and date columns back in.
    dataout[list(non_numeric_cols_syn.columns)] = non_numeric_cols_syn
    dataout[list(placeholder_cols_syn)] = placeholder_cols_syn

    if dataout[eligible_use_cols].isnull().values.any():
        raise ValueError('In the firstorderdiff cleaner, one of the use_cols has a NaN in it after it was processed.')

    # Pass back values in the same column ordered as they came in
    return dataout[list(datain.columns)]

# ----------- END percentilecleaner function. -----------


def atmprcleaner(datain, tol=0.1):

    '''Clean atmospheric pressure values by checking if they are out by more than tol (fraction) from the standard atmospheric pressure. If so, interpolate them.'''
    
    ll = (1-tol)*101325
    ul = (1+tol)*101325

    dataout = datain.mask(np.logical_or(datain <= ll, datain >= ul), other=np.NaN)
    
    dataout = dataout.interpolate(method='linear')
    dataout.fillna(method='bfill', inplace=True)
    dataout.fillna(method='ffill', inplace=True)

    return datain

# ----------- END atmprcleaner function. -----------
    
    
def solarcleaner(datain, master):

    '''Clean solar values by setting zeros at corresponding times in master
       to zero in the synthetic data. This is a proxy for sunrise, sunset,
       and twilight.'''

    # Using the source data - check to see if there
    # should be sunlight at a given hour. If not,
    # then set corresponding synthetic value to zero.
    # If there is a negative value (usually at sunrise
    # or sunset), set it to zero as well.

    datain = datain.mask(datain <= 0, other=0)
    
    datain = datain.interpolate(method='linear')
    datain.fillna(method='bfill', inplace=True)
    datain.fillna(method='ffill', inplace=True)

    return datain

    # A potential improvement would be to calculate sunrise and sunset
    # independently since that is an almost deterministic calculation.

# ----------- END solarcleaner function. -----------


def rhcleaner(rh):

    '''RH values cannot be more than 100 or less than 0.'''

    rhout = pd.DataFrame(rh)

    rhout = rhout.mask(rhout >= 99, other=np.NaN).mask(
        rhout <= 10, other=np.NaN).mask(
        np.isnan(rhout), other=np.NaN)

    rhout = rhout.interpolate(method='linear')
    rhout.fillna(method='bfill', inplace=True)
    rhout.fillna(method='ffill', inplace=True)

    return np.squeeze(rhout.values)

# ----------- END rhcleaner function. -----------


def tdpcleaner(tdp, tdb):

    if not isinstance(tdp, pd.DataFrame):
        tdpout = pd.DataFrame(tdp)

    else:
        tdpout = tdp

    tdpout = tdpout.mask(np.squeeze(tdp) >= np.squeeze(tdb),
                         other=np.NaN)
    tdpout = tdpout.mask(
        np.logical_or(np.squeeze(tdp) >= 50, np.squeeze(tdp) <= -50),
        other=np.NaN)

    if ((np.isnan(tdpout.values)).any()):
        tdpout = tdpout.interpolate(method='linear')
        tdpout = tdpout.fillna(method='bfill').fillna(method='ffill')

    return np.squeeze(tdpout.values)

# ----------- END tdpcleaner function. -----------


def wstats(datain, key, stat):

    grouped_data = datain.groupby(key)

    if stat is 'mean':
        dataout = grouped_data.mean()
    elif stat is 'sum':
        dataout = grouped_data.sum()
    elif stat is 'max':
        dataout = grouped_data.max()
    elif stat is 'min':
        dataout = grouped_data.min()
    elif stat is 'std':
        dataout = grouped_data.std()
    elif stat is 'q1':
        dataout = grouped_data.quantile(0.25)
    elif stat is 'q3':
        dataout = grouped_data.quantile(0.75)
    elif stat is 'med':
        dataout = grouped_data.median()

    return dataout

# ----------- END wstats function. -----------


def calc_rh(tdb, tdp):

    rhout = 100 * (((112 - (0.1 * tdb) + tdp) / (112 + (0.9 * tdb))) ** 8)
    return rhcleaner(rhout)


def calc_tdp(tdb, rh):

    '''Calculate dew point temperature using dry bulb temperature
       and relative humidity.'''

    # Change relative humidity to fraction.
    phi = rh/100

    # Remove weird values.
    phi[phi > 1] = 1
    phi[phi < 0] = 0

    # Convert tdb to Kelvin.
    if any(tdb < 200):
        tdb_k = tdb + 273.15
    else:
        tdb_k = tdb

    # Equations for calculating the saturation pressure
    # of water vapour, taken from ASHRAE Fundamentals 2009.
    # (Eq. 5 and 6, Psychrometrics)

    # This is to distinguish between the two versions of equation 5.
    ice = tdb_k <= 273.15
    not_ice = np.logical_not(ice)

    lnp_ws = np.zeros(tdb_k.shape)

    # Eq. 5, pg 1.2
    lnp_ws[ice] = (
        FROZEN_CONST[0]/tdb_k[ice] + FROZEN_CONST[1] +
        FROZEN_CONST[2]*tdb_k[ice] + FROZEN_CONST[3]*tdb_k[ice]**2 +
        FROZEN_CONST[4]*tdb_k[ice]**3 + FROZEN_CONST[5]*tdb_k[ice]**4 +
        FROZEN_CONST[6]*np.log(tdb_k[ice]))

    # Eq. 6, pg 1.2
    lnp_ws[np.logical_not(ice)] = (
        LIQUID_CONST[0]/tdb_k[not_ice] + LIQUID_CONST[1] +
        LIQUID_CONST[2]*tdb_k[not_ice] +
        LIQUID_CONST[3]*tdb_k[not_ice]**2 +
        LIQUID_CONST[4]*tdb_k[not_ice]**3 +
        LIQUID_CONST[5]*np.log(tdb_k[not_ice]))

    # Temperature in the above formulae must be absolute,
    # i.e. in Kelvin

    # Continuing from eqs. 5 and 6
    p_ws = np.e**(lnp_ws)  # [Pa]

    # Eq. 24, pg 1.8
    p_w = (phi * p_ws) / 1000  # [kPa]

    # Constants for Eq. 39
    EQ39_CONST = [6.54, 14.526, 0.7389, 0.09486, 0.4569]

    p_w[p_w <= 0] = 1e-6
    alpha = pd.DataFrame(np.log(p_w))
    alpha = alpha.replace(
        [np.inf, -np.inf], np.NaN).interpolate(method='linear')

    # Eq. 39
    tdp = alpha.apply(
        lambda x: EQ39_CONST[0] + EQ39_CONST[1]*x + EQ39_CONST[2]*(x**2) +
        EQ39_CONST[3]*(x**3) + EQ39_CONST[4]*(p_w**0.1984))

    # Eq. 40, TDP less than 0°C and greater than -93°C
    tdp_ice = tdp < 0
    tdp[tdp_ice] = 6.09 + 12.608*alpha[tdp_ice] + 0.4959*(alpha[tdp_ice]**2)

    tdp = tdp.replace(
        [np.inf, -np.inf], np.NaN).interpolate(method='linear')

    tdp = tdp.fillna(method='bfill').fillna(method='ffill')

    # tdp = (tdp).rename('tdp')

    tdp = tdpcleaner(tdp, tdb)

    return tdp

# ----------- END tdb2tdp function. -----------


def w2rh(w, tdb, ps=101325):

    if any(tdb < 200):
        tdb_k = tdb + 273.15
    else:
        tdb_k = tdb

    # Humidity ratio W, [unitless fraction]
    # Equation (22), pg 1.8
    p_w = ((w / 0.621945) * ps) / (1 + (w / 0.621945))

    # This is to distinguish between the two versions of equation 5.
    ice = tdb_k <= 273.15
    not_ice = np.logical_not(ice)

    lnp_ws = np.zeros(tdb_k.shape)

    # Eq. 5, pg 1.2
    lnp_ws[ice] = (
        FROZEN_CONST[0] / tdb_k[ice] + FROZEN_CONST[1] +
        FROZEN_CONST[2] * tdb_k[ice] + FROZEN_CONST[3] * tdb_k[ice]**2 +
        FROZEN_CONST[4] * tdb_k[ice]**3 + FROZEN_CONST[5] * tdb_k[ice]**4 +
        FROZEN_CONST[6] * np.log(tdb_k[ice]))

    # Eq. 6, pg 1.2
    lnp_ws[np.logical_not(ice)] = (
        LIQUID_CONST[0] / tdb_k[not_ice] + LIQUID_CONST[1] +
        LIQUID_CONST[2] * tdb_k[not_ice] +
        LIQUID_CONST[3] * tdb_k[not_ice]**2 +
        LIQUID_CONST[4] * tdb_k[not_ice]**3 +
        LIQUID_CONST[5] * np.log(tdb_k[not_ice]))

    # Temperature in the above formulae must be absolute,
    # i.e. in Kelvin

    # Continuing from eqs. 5 and 6
    p_ws = np.e**(lnp_ws)  # [Pa]

    phi = p_w / p_ws  # [Pa] Formula(24), pg 1.8

    rh = phi * 100

    # Relative Humidity from fraction to percentage.
    return rhcleaner(rh)

# ----------- END w2rh function. -----------



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

# Small function to check indices are within bounds
def sanitise_indices(x, size, days_in_block = None, min_val = None):
    # This check ensures that not only will the index chosen fall inside the array but also that the entire block will
    # This ensures that days from the end of the array do not wrap around to the start, ignore days_in_block parameter to disable
    if days_in_block is not None:
        max_size = size - days_in_block + 1
    else:
        max_size = size

    if min_val is None:
        min_val = 0

    if x < 0:
        y = abs(x)
    elif x >= max_size:
        y = max_size - (x - max_size) - 1
    elif x < min_val:
        # This will only occur if the first two conditions are False
        y = min_val
    else:
        y = x

    return y


def ensure_full_year(xy_train, tol=0.1):

    # If the incoming years have missing data, deal with that first.
    # The condition for now tolerates x% missing values.
    tol = 0.1
    if (xy_train.shape[0] % 8760) != 0:
        for year in xy_train.index.year.unique():
            this_year = xy_train.loc[xy_train['year']==year, :]
            if this_year.shape[0] == 8760:
                continue
            elif this_year.shape[0] < ((1-tol)*8760):
                print('year {} cut'.format(year))
                xy_train = xy_train.loc[xy_train['year']!=year, :]
            else:
                print('year {} resampled'.format(year))
                xy_temp = this_year.resample('1H').interpolate(method='linear')
                xy_train = pd.concat([xy_train.loc[xy_train['year']!=year, :], xy_temp], sort=True)

    return xy_train


def dd(xin, bp):

    dd = (xin - bp)
    cdd = dd[dd>0].resample('1Y').sum()
    hdd = np.abs(dd[dd<0].resample('1Y').sum())

    return hdd, cdd

