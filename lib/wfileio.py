import os
import numpy as np
import pandas as pd
import csv
import re
from scipy import interpolate
import math

import petites as petite

'''
This file contains functions to:
    1. load weather data from 'typical' and 'actual' (recorded) weather
       data files.
    2. Write out synthetic weather data to EPW or ESPr weather file formats.
    3. Associated helper functions.
'''

__author__ = 'Parag Rastogi'

# ISSUES TO ADDRESS
# 1. Harmonize WMO numbers - if the incoming number is 5 digits,
# add leading zero (e.g., Geneva)
# 2. Implement something to convert GHI to DNI and DHI.
# Maybe use Erbs model like before.

# %%

# Useful strings and constants.

# List of keywords that identify TMY and AMY files.
keywords = dict(tmy=('nrel', 'iwec', 'ishrae', 'cwec',
                     'igdg', 'tmy3', 'meteonorm'),
                amy=('ncdc', 'nsrdb', 'nrel_indiasolar',
                     'ms', 'WY2', 'nasa_saudi'))
wformats = ('epw', 'espr', 'csv', 'fin4')

# List of values that could be NaNs.
nanlist = ('9900', '-9900', '9999', '99', '-99', '9999.9', '999.9', ' ', '-')

# A generic ESPR header that can be used in a print or str
# command with format specifiers.
espr_generic_header = '''*CLIMATE
# ascii weather file from {0},
# defined in: {1}
# col 1: Diffuse solar on the horizontal (W/m**2)
# col 2: External dry bulb temperature   (Tenths DEG.C)
# col 3: Direct normal solar intensity   (W/m**2)
# col 4: Prevailing wind speed           (Tenths m/s)
# col 5: Wind direction     (clockwise deg from north)
# col 6: Relative humidity               (Percent)
{2}               # site name
 {3},{4},{5},{6}   # year, latitude, long diff, direct normal rad flag
 {7},{8}    # period (julian days)'''

# # The standard columns used by indra.
# std_cols = ('year', 'month', 'day', 'hour', 'tdb', 'tdp', 'rh',
#             'ghi', 'dni', 'dhi', 'wspd', 'wdir')

std_cols = {
    'epw':['year', 'month', 'day', 'hour', 'minute', 'qualflags',
              'tdb', 'tdp', 'rh', 'atmpr', 'etrh', 'etrn', 'hir',
              'ghi', 'dni', 'dhi', 'ghe', 'dne', 'dhe', 'zl',
              'wdir', 'wspd', 'tsky', 'osky', 'vis', 'chgt',
              'pwo', 'pwc', 'pwt', 'aopt', 'sdpt', 'slast'],
    'espr':['dhi', 'tdb', 'dni', 'wspd', 'wdir', 'rh'],
    'fin4': ['year', 'month', 'day', 'hour',
               'tdb', 'tdp', 'atmpr', 'sky', 'osky',
               'wspd', 'wdir', 'ghi', 'dni', 'Pres',
               'Rain', 'vis', 'chgt', 'solarz']}

# %%

def get_weather(stcode, fpath):

    # This function calls the relevant reader based on the file_type.

    # Initialise as a non-object.
    wdata = None
    locdata = None
    header = None

    file_type = os.path.splitext(fpath)[-1].replace('.', '').lower()
    
    if not os.path.isfile(fpath):
        print('I cannot find file {0}.'.format(fpath) +
              ' Returning empty dataframe.\r\n')
        return wdata, locdata, header

    # Load data for given station.

    if file_type == 'pickle' or file_type == 'p':
        try:
            wdata_array = pd.read_pickle(fpath)
            return wdata_array
        except Exception as err:
            print('You asked me to read a pickle but I could not. ' +
                  'Trying all other formats.\r\n')
            print('Error: ' + str(err))
            wdata = None
            header = None
            locdata = None

    elif file_type == 'epw':

        try:
            wdata, locdata, header = read_epw(fpath)
        except Exception as err:
            print('Error: ' + str(err))
            wdata = None
            header = None
            locdata = None

    elif file_type == 'espr':

        wdata, locdata, header = read_espr(fpath)

    elif file_type == 'csv':

        try:
            wdata = pd.read_csv(fpath, header=0)
            wdata.columns = ['year', 'month', 'day', 'hour', 'tdb', 'tdp', 'rh',
                             'ghi', 'dni', 'dhi', 'wspd', 'wdir']
            # Location data is nonsensical, except for station code,
            # which will be reassigned later in this function.
            locdata = dict(loc=stcode, lat='00', long='00',
                           tz='00', alt='00', wmo='000000')
            header = ('# Unknown incoming file format ' +
                      '(not epw or espr)\r\n' +
                      '# Dummy location data: ' +
                      'loc: {0}'.format(locdata['loc']) +
                      'lat: {0}'.format(locdata['lat']) +
                      'long: {0}'.format(locdata['long']) +
                      'tz: {0}'.format(locdata['tz']) +
                      'alt: {0}'.format(locdata['alt']) +
                      'wmo: {0}'.format(locdata['wmo']) +
                      '\r\n')

        except Exception as err:
            print('Error: ' + str(err))
            wdata = None
            header = None
            locdata = None

    elif file_type == 'fin4':

#        try:
            wdata, locdata, header = read_fin4(fpath)
#        except Exception as err:
#            print('Error: ' + str(err))
#            wdata = None
#            header = None
#            locdata = None

    # End file_type if statement.

    if locdata is not None:
        locdata['loc'] = stcode

    if wdata is None:
        print('I could not read the file you gave me with the format ' +
              'you specified. Trying all readers.\r\n')
    else:
    
        if len(wdata.index.year.unique()) == 1:
            # Ensure the index is hourly.
            # Save the text columns first.
            wdata_text_col = wdata.select_dtypes(include=object, exclude=np.number)
            wdata_text_col = wdata_text_col.resample('1H').first().sort_index(axis=0, ascending=True)
            # Then take the means of the numeric columns.
            wdata = wdata.resample('1H').mean()
            # Interpolate resulting nans.
            wdata.interpolate(method = 'linear', limit = 6, limit_area = None, inplace = True)

            # Re-attach the text columns and re-arrange to standard order.
            wdata = pd.concat((wdata, wdata_text_col), axis=1)        
        else:
            wdata = wdata.dropna(how='all')
            wdata = wdata.sort_values(by=['month','day','hour'])
            wdata = petite.remove_leap_day(wdata)
            remake_index(wdata)

        if file_type in ['epw', 'espr']:
            if (wdata.columns != std_cols[file_type]).any():
                wdata = wdata[std_cols[file_type]]
        elif file_type == 'fin4':
            if (wdata.columns != std_cols[file_type] + ['rh']).any():
                wdata = wdata[std_cols[file_type] + ['rh']]

        wdata = petite.remove_leap_day(wdata)
        
    return wdata, locdata, header

# %%


def read_fin4(fpath):

    locdata = dict()
    hlines = 3
    header = list()

    with open(fpath, 'r') as openfile:
        for ln in range(0, hlines):
            header.append(openfile.readline())

    wdata = pd.read_csv(
        fpath, sep='\s+', skiprows=[0, 1, 2],
        names=std_cols['fin4'], dtype=str, index_col=False)

    wdata = wdata.dropna(axis=1, how='all')
    
    for col in wdata.columns:
        try:
            wdata.loc[:, col] = wdata[col].apply(
                lambda x: float(''.join(re.findall('[0-9.-]', x))))
        except Exception:
            wdata.loc[:, col] = np.repeat(np.nan, wdata[col].shape)

        if col in ['year', 'month', 'day', 'hour']:
            wdata[col] = wdata[col].apply(lambda x: int(x))
            
    if len(wdata['year'].unique()) > 1:
        wdata['year'] = 2223
    wdata.index = pd.DatetimeIndex(start='{}-01-01 00:00:00'.format(wdata['year'].unique()[0]),
                                   end='{}-12-31 23:00:00'.format(wdata['year'].unique()[0]),
                                   freq='1H')

    missing_cols = [x for x in std_cols['fin4'] if x not in wdata.columns]
    for col in missing_cols:
        wdata[col] = 0

    wdata['rh'] = pd.Series(petite.calc_rh(wdata['tdb'], wdata['tdp']),
                            index=wdata.index)
    
    # Convert mbar to Pa
    wdata['atmpr'] = petite.check_pressure_unit(wdata['atmpr'], 'mbar')

    remake_index(wdata)

    return wdata, locdata, header


# %%
# Number of days in each month.
m_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def day_of_year(month, day):

    month = month.astype(int) - 1
    doy = np.zeros_like(day, dtype=int)

    for m, mon in enumerate(month):
        doy[m] = (day[m] + np.sum(m_days[0:mon])).astype(int)

    return doy

# End function day_of_year
# %%


def day_of_month(day):

    month = np.zeros_like(day, dtype=int)
    dom = np.zeros_like(day, dtype=int)

    for d, doy in enumerate(day):

        rem = doy
        prev_ndays = 0

        for m, ndays in enumerate(m_days):
            # The iterator 'm' starts at zero.

            if rem <= 0:
                # Iterator has now reached the incomplete month.

                # The previous month is the correct month.
                # Not subtracting 1 because the iterator starts at 0.
                month[d] = m

                # Add the negative remainder to the previous month's days.
                dom[d] = rem + prev_ndays
                break

            # Subtract number of days in this month from the day.
            rem -= ndays
            # Store number of days from previous month.
            prev_ndays = ndays

    return month, dom

# End function day_of_month


# %%
epw_colnames = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'QualFlags',
                'TDB', 'TDP', 'RH', 'ATMPR', 'ETRH', 'ETRN', 'HIR',
                'GHI', 'DNI', 'DHI', 'GHE', 'DNE', 'DHE', 'ZL',
                'WDIR', 'WSPD', 'TSKY', 'OSKY', 'VIS', 'CHGT',
                'PWO', 'PWC', 'PWT', 'AOPT', 'SDPT',
                'SLAST', 'UnknownVar1', 'UnknownVar2', 'UnknownVar3']


def read_epw(fpath, epw_colnames=epw_colnames):

    # Names of the columns in EPW files. Usually ignore the last
    # three columns.

    # Number of header lines expected.
    hlines = 8

    # Convert the names to lowercase.
    epw_colnames = [x.lower() for x in epw_colnames]

    # Read table, ignoring header lines.
    wdata = pd.read_csv(fpath, delimiter=',', skiprows=hlines,
                        header=None, names=epw_colnames,
                        index_col=False)
    
    remake_index(wdata)

    if len(wdata.columns) == 35:
        # Some files have three extra columns
        # (usually the TMY files from USDOE).
        # Delete those columns if found.
        wdata = wdata.drop(['unknownvar1', 'unknownvar2',
                            'unknownvar3'], axis=1)
    
    for col in wdata.columns:
        
        if isinstance(type(wdata[col][0]), str):
            wdata[col] = wdata[col].apply(lambda x: (x.strip()))
        
        if col in ['year', 'month', 'day', 'hour']:
            wdata[col] = wdata[col].astype('int32', errors='ignore')
        else:
            wdata[col] = wdata[col].astype('float', errors='ignore')

    # Read header and assign all metadata.
    header = list()
    hf = open(fpath, 'r')
    for ln in range(0, hlines):
        header.append(hf.readline())

    infoline = (header[0].strip()).split(',')

    locdata = dict(loc=infoline[1], lat=infoline[6], long=infoline[7],
                   tz=infoline[8], alt=infoline[9], wmo=infoline[5])

    if wdata.empty:

        print('Could not locate a file with given station name.' +
              ' Returning empty table.\r\n')

    return wdata, locdata, header

# ----------- END read_epw function -----------
# %%


def read_espr(fpath):

    # Missing functionality - reject call if path points to binary file.

    # Uniform date index for all tmy weather data tables.
    dates = pd.date_range('1/1/2223', periods=8760, freq='H')

    fpath_fldr, fpath_name = os.path.split(fpath)
    sitename = fpath_name.split(sep='.')
    sitename = sitename[0]

    with open(fpath, 'r') as f:
        content = f.readlines()

    hlines = 12

    # Split the contents into a header and body.
    header = content[0:hlines]

    # Find the year of the current file.
    yline = [line for line in header if 'year' in line]

    yline = yline[0].split('#')[0]

    if ',' in yline:
        yline_split = yline.split(',')
    else:
        yline_split = yline.split()
    year = yline_split[0].strip()

    locline = [line for line in header if ('latitude' in line)][0]
    siteline = [line for line in header if ('site name' in line)][0]

    if ',' in locline:
        locline = locline.split(',')
    else:
        locline = locline.split()

    if ',' in siteline:
        siteline = siteline.split(',')
    else:
        siteline = siteline.split()

    locdata = dict(loc=siteline[0], lat=locline[1], long=locline[2],
                   tz='00', alt='0000', wmo='000000')
    # ESP-r files do not contain timezone, altitude, or WMO number.

    body = content[hlines:]

    del content

    # Find the lines with day tags.
    daylines = [[idx, line] for [idx, line] in enumerate(body)
                if 'day' in line]

    dataout = np.zeros([8760, 11])

    dcount = 0

    for idx, day in daylines:

        # Get the next 24 lines.
        daylist = np.asarray(body[idx+1:idx+25])

        # Split each line of the current daylist into separate strings.
        if ',' in daylist[0]:
            splitlist = [element.split(',') for element in daylist]
        else:
            splitlist = [element.split() for element in daylist]

        # Convert each element to a integer, then convert the resulting
        # list to a numpy array.
        daydata = np.asarray([list(map(int, x)) for x in splitlist])

        # Today's time slice.
        dayslice = range(dcount, dcount+24, 1)

        # This will split the day-month header line on the gaps.
        if ',' in day:
            splitday = day.split(',')
        else:
            splitday = day.split(' ')

        # Remove blanks.
        splitday = [x for x in splitday if x != '']
        splitday = [x for x in splitday if x != ' ']

        # Month.
        dataout[dayslice, 0] = np.repeat(int(splitday[-1]), len(dayslice))

        # Day of month.
        dataout[dayslice, 1] = np.repeat(int(splitday[2]), len(dayslice))

        # Hour (of day).
        dataout[dayslice, 2] = np.arange(0, 24, 1)

        # tdb, input is in deci-degrees, convert to degrees.
        dataout[dayslice, 3] = daydata[:, 1]/10

        # tdp is calculated after this loop.

        # rh, in percent.
        dataout[dayslice, 5] = daydata[:, 5]

        # ghi is calculated after this loop.

        # dni, in W/m2.
        dataout[dayslice, 7] = daydata[:, 2]

        # dhi, in W/m2.
        dataout[dayslice, 8] = daydata[:, 0]

        # wspd, input is in deci-m/s.
        dataout[dayslice, 9] = daydata[:, 3]/10

        # wdir, clockwise deg from north.
        dataout[dayslice, 10] = daydata[:, 4]

        dcount += 24

    # tdp, calculated from tdb and rh.
    dataout[:, 4] = petite.calc_tdp(dataout[:, 3], dataout[:, 5])

    # ghi, in W/m2.
    dataout[:, 6] = dataout[:, 7] + dataout[:, 8]

    # wspd can have bogus values (999)
    dataout[dataout[:, 10] >= 999., 10] = np.nan
    idx = np.arange(0, dataout.shape[0])
    duds = np.logical_or(np.isinf(dataout[:, 10]), np.isnan(dataout[:, 10]))
    int_func = interpolate.interp1d(
        idx[np.logical_not(duds)], dataout[np.logical_not(duds), 10],
        kind='nearest', fill_value='extrapolate')
    dataout[duds, 10] = int_func(idx[duds])

    dataout = np.concatenate((np.reshape(np.repeat(int(year), 8760),
                                         [-1, 1]), dataout), axis=1)

    clmdata = pd.DataFrame(data=dataout, index=dates,
                           columns=['year', 'month', 'day', 'hour',
                                    'tdb', 'tdp', 'rh',
                                    'ghi', 'dni', 'dhi', 'wspd', 'wdir'])

    return clmdata, locdata, header

# ----------- END read_espr function -----------
# %%


def give_weather(df, locdata=None, stcode=None, header=None,
                 masterfile='GEN_IWEC.epw',
                 path_file_out='.', incoming_cols=None):

    file_type = os.path.splitext(path_file_out)[-1].replace('.', '').lower()
    
    df.dropna(how='all', axis=1, inplace=True)

    # If no columns were passed, infer them from the columns of the dataframe.
    if incoming_cols is None:
        incoming_cols = df.columns

    if file_type in ['epw', 'espr']:
        if (df.columns != std_cols[file_type]).any():
            df = df[std_cols[file_type]]
    elif file_type == 'fin4':
        if (df.columns != std_cols[file_type] + ['rh']).any():
            df = df[std_cols[file_type] + ['rh']]

    # Check if incoming temperature values are in Kelvin.
    for col in ['tdb', 'tdp']:
        if any(df.loc[:, col] > 200):
            df.loc[:, col] = df.loc[:, col] - 273.15

    success = False

    year = np.unique(df.index.year)[0]

    # Convert date columns to integers.
    if 'month' in df.columns:
        df.loc[:, 'month'] = df.loc[:, 'month'].astype(int) 
    if 'day' in df.columns:
        df.loc[:, 'day'] = df.loc[:, 'day'].astype(int) 
    if 'hour' in df.columns:
        df.loc[:, 'hour'] = df.loc[:, 'hour'].astype(int) 

    # If last hour was interpreted as first hour of next year, you might
    # have two years.
    # This happens if the incoming file has hours from 1 to 24.
    if isinstance(year, list):
        counts = np.bincount(year)
        year = np.argmax(counts)

    if path_file_out == '.':
        # Make a standardised name for output file.
        filepath = os.path.join(
            path_file_out, 'wf_out_{0}_{1}'.format(
                np.random.randint(0, 99, 1), year))
    else:
        # Files need to be renamed so strip out the extension.
        filepath = path_file_out.replace(
            '.a', '').replace('.epw', '').replace('.csv', '')
        
    if file_type == 'espr':

        # These columns will be replaced.
        esp_columns = ['dhi', 'tdb', 'dni', 'wspd', 'rh']

        if filepath[-2:] != '.a':
            filepath = filepath + '.a'

        esp_master, locdata, header = read_espr(masterfile)

        # Replace the year in the header.
        yline = [line for line in header if 'year' in line]
        yval = yline[0].split(',')
        yline[0] = yline[0].replace(yval[0], str(year))
        header = [yline[0] if 'year' in line else line
                  for line in header]
        # Cut out the last new-line character since numpy savetxt
        # puts in a newline character after the header anyway.
        header[-1] = header[-1][:-1]

        for col in esp_columns:
            esp_master.loc[:, col] = df[col].values
            if col in ['tdb', 'wspd']:
                # Deci-degrees and deci-m/s respectively.
                esp_master.loc[:, col] *= 10
        # Create a datetime index for this year.
        esp_master.index = pd.DatetimeIndex(
            start='{:04d}-01-01 00:00:00'.format(year),
            end='{:04d}-12-31 23:00:00'.format(year),
            freq='1H')

        # Save month and day to write out to file as separate rows.
        monthday = (esp_master.loc[:, ['day', 'month']]).astype(int)

        # Drop those columns that will not be written out.
        esp_master = esp_master.drop(
            ['year', 'month', 'day', 'hour', 'ghi', 'tdp'],
            axis=1)
        # Re-arrange the columns into the espr clm file order.
        esp_master = esp_master[esp_columns]
        # Convert all data to int.
        esp_master = esp_master.astype(int)

        master_aslist = esp_master.values.tolist()

        for md in range(0, monthday.shape[0], 25):
            md_list = [str('* day {0} month {1}'.format(
                monthday['day'][md], monthday['month'][md]))]
            master_aslist.insert(md, md_list)

        # Write the header to file - though the delimiter is
        # mostly meaningless in this case.
        with open(filepath, 'w') as f:
            spamwriter = csv.writer(f, delimiter='\n', quotechar='',
                                    quoting=csv.QUOTE_NONE,
                                    escapechar=' ',
                                    lineterminator='\n')
            spamwriter.writerow([''.join(header)])

            spamwriter = csv.writer(f, delimiter=',', quotechar='',
                                    quoting=csv.QUOTE_NONE,
                                    escapechar=' ',
                                    lineterminator='\n ')
            for line in master_aslist[:-1]:
                spamwriter.writerow(line)

            spamwriter = csv.writer(f, delimiter=',', quotechar='',
                                    quoting=csv.QUOTE_NONE,
                                    lineterminator='\n\n')
            spamwriter.writerow(master_aslist[-1])

        if os.path.isfile(filepath):
            success = True
        else:
            success = False

        # End espr writer.

    elif file_type == 'epw':

        if filepath.split('.')[-1] != 'epw':
            filepath = filepath + '.epw'

        epw_fmt = ['%-4u', '%-1u', '%-1u', '%-1u', '%-1u', '%-44s'] + (np.repeat('%-3.2f', len(epw_colnames) - (6 + 3))).tolist()

        epw_master, locdata, header = read_epw(masterfile)
        # Cut out the last new-line character since numpy savetxt
        # puts in a newline character after the header anyway.
        header[-1] = header[-1][:-1]

        # These columns will be replaced.
        epw_columns = ['tdb', 'tdp', 'rh', 'ghi', 'dni', 'dhi', 'wspd', 'wdir']
        for col in epw_columns:
            epw_master.loc[:, col] = df[col].values

        # Replace the year of the master file.
        epw_master['year'] = year
        
        epw_master = epw_master.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        np.savetxt(filepath, epw_master.values, fmt=epw_fmt,
                   delimiter=',', header=''.join(header),
                   comments='')

        if os.path.isfile(filepath):
            success = True
        else:
            success = False

        # End EPW writer.

    elif file_type == 'fin4':
        if filepath.split('.')[-1] != 'fin4':
            filepath = filepath + '.fin4'

        _, _, header = read_fin4(masterfile)
        
        # Strip the last end-of-line character.
        header[-1] = header[-1].strip('\r').strip('\n')

        # Convert pressure back to millibars.
        df.loc[:, 'atmpr'] = df.loc[:, 'atmpr'] / 100
        
        if 'rh' in df.columns:
            df = df.drop(labels='rh', axis=1)

        df.loc[:, 'tdp'] = petite.tdpcleaner(df['tdp'], df['tdb'])
        
        # Columns that are not changed in fin4 files will be reset to zero.
        # Add a column of None for atmpr inHg to output blanks.
        df.loc[:, 'atmprinHg'] = None
        df.loc[:, 'sky'] = 99
        df.loc[:, 'osky'] = 99
        df.loc[:, 'Pres'] = 999
        df.loc[:, 'Rain'] = 9999
        df.loc[:, 'vis'] = 99999
        df.loc[:, 'chgt'] = 999999
        
        # Reorder columns so iniHg is in the right place.
        df_columns = df.columns[:7].values.tolist()
        df_columns.append(df.columns[-1])
        df_columns.extend(df.columns[7:-1])
        df = df[df_columns]

        # Interpolate any nans still hanging around.
        df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')       
        df['wdir'] = (np.round(df['wdir'], decimals = 0)).astype(int)
        
        # df.to_csv(filepath, sep=' ', header=' '.join(header), index=False)
        fin_fmt, fin_fmt_write = auto_formatter(df)
                
        output_writer = open(filepath, 'w')
        for i in range(-1, df.shape[0]):
            if i < 0:
                to_write = ''.join(header) + '\n'
            else:
                to_write = fin_fmt_write.format(*[v if v is not None else ' ' for v in df.iloc[i, :].tolist()]) + '\n'
            
            output_writer.write(to_write)
        output_writer.close()

        if os.path.isfile(filepath):
            success = True
        else:
            success = False

    else:

        if filepath.split('.')[-1] != 'csv':
            filepath = filepath + 'csv'

        df.to_csv(filepath, sep=',', header=True, index=False)

        if os.path.isfile(filepath):
            success = True
        else:
            success = False

    if success:
        print('Write success.')
    else:
        print('Some error prevented file from being written.')

# ----------- End give_weather function. -----------


def remake_index(wdata):
    # Only for TMY files - not required for prooduction as we will always have historical data
    year = wdata['year'].unique()

    if len(year) > 1:  # isinstance(year, list):  # isinstance list doesn't work because output of above command is a numpy array.
        print('Found multiple years in file name, assigning 2223.')
        original_years = wdata.loc[:, 'year'].dropna()
        print(original_years.unique())
        
        wdata['year'] = 2223

    # Uniform date index for all tmy weather data tables.
    wdata.index = pd.to_datetime(wdata[['year', 'month', 'day', 'hour']])

    if list(wdata['hour'].unique()) == list(range(1, 25)):
        wdata.index = wdata.index - pd.Timedelta(hours=1)


# Function to take data frame and suggest the best file formats for it
def auto_formatter(df, max_dec_places = 1):
    formats = list()
    formats_write = list()
    
    for i, col in enumerate(df):
        # Check if column is entirely None values
        if not df[col].isnull().any():
            # Calculate the number of digits before the point using log10, but set a minimum of 1
            exponand = max(np.absolute(df[col]))
            if exponand == 0:
                exponent = 1
            else:
                exponent = math.ceil(math.log10(exponand))
                        
            # Calculate the number of digits after the point using Python's built in string formatter which takes care of the decimal/float nonsense
            decimal_places = max(df[col].astype(str).apply(lambda x: num_dec_places(x)))
            decimal_places = min(decimal_places, 1)
                        
            # Ascertain the data type
            if df.dtypes[col] in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                decimal_places = 0
                data_type = 'd'
            elif df.dtypes[col] in [np.float16, np.float32, np.float64]:
                data_type = 'f'
               
            if data_type == 'd':
                exponent = max(exponent, 1)
            elif data_type == 'f':
                exponent = max(exponent, 3)            
            
            # Put it all together
            output_format = '%{}'.format(exponent + decimal_places + 1) + ('.{}'.format(decimal_places) if data_type == 'f' else '') + data_type
            output_format_write = '{{:{}{}'.format(' > ' if data_type == 'f' else '>', exponent + decimal_places + 1 if data_type == 'f' else exponent) + ('.{}'.format(decimal_places) if data_type == 'f' else '') + data_type + ('}  ' if i > 3 else '} ')
            
            # Solar z has only one digit before the decimal point.
            if col == 'solarz':
                output_format = '%5.4f'
                output_format_write = '{:>5.4f}'
            # Wind speed has no sign.
            if col == 'wspd':
                minwidth = re.findall('\d\.', output_format)[0][0]
                output_format = output_format.replace(' > ', '>').replace(minwidth, str(int(minwidth)-1))
                output_format_write = output_format_write.replace(' > ', '>').replace(minwidth, str(int(minwidth)-1))
                
            if col == 'ghi' or col == 'dni' or col == 'atmpr':
                minwidth = re.findall('\d\.', output_format)[0][0]
                output_format = output_format.replace(minwidth, str(int(minwidth)+1))
                output_format_write = output_format_write.replace(minwidth, str(int(minwidth)+1))
                
        else:
            output_format = '%s'
            output_format_write = '{:6s} '
        
        formats.append(output_format)       
        formats_write.append(output_format_write)      
        
    # For checking the output
    # [(df.columns[i], f) for i, f in enumerate(formats)]
    return formats, ''.join(formats_write)
        
def num_dec_places(x):
    if len(x.split('.')) > 1:
        return len(x.split('.')[1])
    else:
        return 0
        


















