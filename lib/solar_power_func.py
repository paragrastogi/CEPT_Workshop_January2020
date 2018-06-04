#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 2018

@author: parag rastogi

This function is based on the tmy_to_power.ipynb file available
at github.com/pvlib/pvlib-python/tree/master/docs/tutorials .
The original file was written by Will Holmgren and Rob Andrews.
"""

# built-in python modules
import os
import inspect

# scientific python add-ons
import numpy as np
import pandas as pd

# Import the pvlib library
import pvlib

# Import the file input-output module.
import wfileio as wf


# Load TMY data

# Find the absolute file path to your pvlib installation
pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))

# # Path to a data file
# path_tmy_data = os.path.join(
#    '..', 'ddn', 'IND_UT_Dehradun.421110_ISHRAE2014.epw')

# pvlib uses 0=North, 90=East, 180=South, 270=West convention


def tmy_to_power(path_tmy_data='.', tmy_data=np.NaN,
                 locdata=np.NaN, header='',
                 surface_tilt=30, surface_azimuth=180,
                 albedo=0.2, silent=True):

    if path_tmy_data != '.':
        # read tmy data.
        tmy_data, locdata, header = wf.get_weather(
            'tmy_data', path_tmy_data, file_type='epw')
        tmy_data.index.name = 'Time'
    else:
        if np.all(np.isnan(tmy_data)):
            print('You did not give me either a path to a TMY file or a dataframe.')
            return 0

    # ## Calculate modeling intermediates

    # Before we can calculate power for all times in the
    # weather file, we will need to calculate:
    # * solar position
    # * extra terrestrial radiation
    # * airmass
    # * angle of incidence
    # * POA sky and ground diffuse radiation
    # * cell and module temperatures

    # # First, define some PV system parameters.
    # surface_tilt = 30
    # surface_azimuth = 180
    # albedo = 0.2

    # create pvlib Location object based on meta data
    sand_point = pvlib.location.Location(
        np.double(locdata['lat']),
        np.double(locdata['long']),
        tz='Asia/Calcutta',
        altitude=np.double(locdata['alt']),
        name='tmy_data')

    # ### Solar position

    # Calculate the solar position for all times in the TMY file.
    #
    # The default solar position algorithm is based on Reda
    # and Andreas (2004). Our implementation is pretty fast,
    # but you can make it even faster if you install [``numba``]
    # (http://numba.pydata.org/#installing) and use add
    # ``method='nrel_numba'`` to the function call below.
    solpos = pvlib.solarposition.get_solarposition(
        tmy_data.index, sand_point.latitude,
        sand_point.longitude)

    # The funny looking jump in the azimuth is just due to
    # the coarse time sampling in the TMY file.

    # ### DNI ET
    #
    # Calculate extra terrestrial radiation. This is needed
    # for many plane of array diffuse irradiance models.

    # the extraradiation function returns a simple numpy array
    # instead of a nice pandas series. We will change this
    # in a future version
    dni_extra = pvlib.irradiance.extraradiation(tmy_data.index)
    dni_extra = pd.Series(dni_extra, index=tmy_data.index)

    # ### Airmass
    # Calculate airmass. Lots of model options here, see the
    # ``atmosphere`` module tutorial for more details.

    airmass = pvlib.atmosphere.relativeairmass(
        solpos['apparent_zenith'])

    # The funny appearance is due to aliasing and setting
    # invalid numbers equal to ``NaN``.
    # Replot just a day or two and you'll see that the
    # numbers are right.

    # ### POA sky diffuse

    # Use the Hay Davies model to calculate the plane of array
    # diffuse sky radiation. See the ``irradiance`` module
    # tutorial for comparisons of different models.

    poa_sky_diffuse = pvlib.irradiance.haydavies(
        surface_tilt, surface_azimuth,
        tmy_data['dhi'], tmy_data['dni'], dni_extra,
        solpos['apparent_zenith'], solpos['azimuth'])

    # There are some numerical errors in the sky diffuse
    # calculation where the value spikes.
    # Take the quantities that are larger than the ambient
    # sky diffuse radiation and interpolate them.
    poa_sky_diffuse[poa_sky_diffuse > tmy_data.dhi] = np.NaN
    poa_sky_diffuse = poa_sky_diffuse.interpolate(method='time')

    # ### POA ground diffuse
    #
    # Calculate ground diffuse. We specified the albedo above.
    # You could have also provided a string to the
    # ``surface_type`` keyword argument.

    poa_ground_diffuse = pvlib.irradiance.grounddiffuse(
        surface_tilt, tmy_data['ghi'], albedo=albedo)

    # ### AOI
    #
    # Calculate AOI
    aoi = pvlib.irradiance.aoi(
        surface_tilt, surface_azimuth,
        solpos['apparent_zenith'], solpos['azimuth'])

    # Note that AOI has values greater than 90 deg. This is ok.

    # ### POA total
    #
    # Calculate POA irradiance

    poa_irrad = pvlib.irradiance.globalinplane(
        aoi, tmy_data['dni'], poa_sky_diffuse,
        poa_ground_diffuse)

    # ### Cell and module temperature
    #
    # Calculate pv cell and module temperature

    pvtemps = pvlib.pvsystem.sapm_celltemp(
        poa_irrad['poa_global'], tmy_data['wspd'],
        tmy_data['tdb'])

    # ## DC power using SAPM

    # Get module data from the web.
    sandia_modules = pvlib.pvsystem.retrieve_sam(
        name='SandiaMod')

    # Choose a particular module
    sandia_module = sandia_modules.Canadian_Solar_CS5P_220M___2009_

    # Calculate the effective irradiance
    effective_irradiance = \
        pvlib.pvsystem.sapm_effective_irradiance(
            poa_irrad.poa_direct, poa_irrad.poa_diffuse,
            airmass, aoi, sandia_module)

    # Run the SAPM using the parameters we calculated above.
    sapm_out = pvlib.pvsystem.sapm(
        effective_irradiance, pvtemps.temp_cell, sandia_module)

    # DC power using single diode.
    cec_modules = pvlib.pvsystem.retrieve_sam(name='CECMod')
    cec_module = cec_modules.Canadian_Solar_CS5P_220M

    photocurrent, saturation_current, \
        resistance_series, resistance_shunt, nNsVth = \
        pvlib.pvsystem.calcparams_desoto(
            poa_irrad.poa_global,
            temp_cell=pvtemps['temp_cell'],
            alpha_isc=cec_module['alpha_sc'],
            module_parameters=cec_module,
            EgRef=1.121, dEgdT=-0.0002677)

    single_diode_out = pvlib.pvsystem.singlediode(
        photocurrent, saturation_current,
        resistance_series, resistance_shunt, nNsVth)

    sapm_inverters = pvlib.pvsystem.retrieve_sam(
        'sandiainverter')

    # Choose a particular inverter.
    sapm_inverter = sapm_inverters[
        'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    p_acs = pd.DataFrame()
    p_acs['sapm'] = pvlib.pvsystem.snlinverter(
        sapm_out.v_mp, sapm_out.p_mp, sapm_inverter)
    p_acs['sd'] = pvlib.pvsystem.snlinverter(
        single_diode_out.v_mp, single_diode_out.p_mp,
        sapm_inverter)

    mask = p_acs['sapm'] < 0
    p_acs.loc[mask, 'sapm'] = 0

    mask = p_acs['sd'] < 0
    p_acs.loc[mask, 'sd'] = 0

    # Some statistics on the AC power

    if not silent:
        print(p_acs.describe())
        print(p_acs.sum())

    return p_acs
