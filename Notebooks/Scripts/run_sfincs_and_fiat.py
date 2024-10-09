# Compound_TGP
# Copyright (C) 2024 Terlinden-Ruhl

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import subprocess
import time
from pathlib import Path

import fiat
import geopandas as gpd
import hydromt
import hydromt_sfincs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from hydromt.log import setuplog
from hydromt_sfincs import SfincsModel, utils
from scipy.signal import argrelextrema

import Notebooks.Scripts.Useful as use
from Notebooks.Scripts.sfincs_utils import run_sfincs

## Script contains functions to run coupled SFINCS and FIAT models

def run_simulation(index, folder_to_save, template_folder, sfincs_model, s_mag, s_dur, p_mag,
                   p_dur, tide_mag = 0.98, p_lag = 0, base_flow = 200, plot = False, artificial_tide = False, sub_county = None):
    '''
    Parameters
    ------
    index: int
        identifier of simulation that is going to be run
    folder_to_save: str
        name of the folder that will be saved in the "Models" folder
    template_folder: str
        relative path to the SFINCS folder from FloodAdapt
    sfincs_model: obj
        Object created when initializing the sfincs model
    s_mag: int
        skew surge magnitude in m
    s_dur: int
        duration of skew surge in number of tidal cycles
    p_mag: int
        precipitation magnitude in mm/hr
    p_dur: int
        duration of precipitation in hours
    tide_mag: int
        tidal magnitude, default is mean high water spring tide (from NOAA)
    p_lag: int
        precipitation lag with respect to high tide in hours
    base_folw: int
        constant discharge used in m^3/s. Defulat is the mean discharge
    plot: bool
        plots the boundary conditions. Default is False
    artificial_tide: bool
        if False will sample a historical tide with magnitude: t_mag. If True will generate an artifical signal with an M2 tidal period.
    sub_county: str
        default is none, will collect total economic damages. If a sub-county is provided it will collect the damages related to that sub-county.

    Returns
    ------
    t_dam: float
        Economic damages for chosen sub-county
    times: arr like
        Numpy contains the time it took for SFINCS and FIAT to run
    '''

    sfincs_folder_path = "Models/" + folder_to_save + "/sfincs_folder/sfincs_" + str(index)
    fiat_folder_path = "Models/" + folder_to_save + "/fiat_folder/fiat_" + str(index)

    meter_feet = 3.28084
    # Create new parent folders 
    use.create_empty_folder("Models/" + folder_to_save)
    use.create_empty_folder("Models/" + folder_to_save + "/sfincs_folder")
    use.create_empty_folder("Models/" + folder_to_save + "/fiat_folder")
    use.create_empty_folder(sfincs_folder_path)

    model_path = Path(sfincs_folder_path)
    sfincs_model.set_root(model_path, mode = 'w+')
    sfincs_model.write()
    span = 7 + 6
    if artificial_tide:
        use.generate_tide(template_folder, sfincs_model, mag = tide_mag, span = span)
        peaks = np.round(np.arange(0,  span * 24 * 6, 12.42 * 6)).astype(int) # 6 is a fudge factor for the resolution (6 timesteps per hour), 7 is the number of days (needs to be the same as tide)
        peak_index = len(peaks)//2
    else:
        peak_ind = use.retrieve_historical_tide(template_folder, sfincs_model, mag = tide_mag, span = span)
    
    tide = use.index_first(sfincs_model.forcing['bzs'].copy())
    time_ind = tide[0].coords['time'].values

    if artificial_tide:
        center = time_ind[peaks][peak_index]
    else:
        center = time_ind[peak_ind]

    sfincs_model.forcing.pop("bzs", None)  # reset
    surge_da, start_s, end_s = use.artificial_driver(time_ind, mag = s_mag, dur = int(12.42 * s_dur), center = center, # assume m2 tidal cycle when setting duration
                                                        name = r'Surge [$m$]', lag = 0, base = 0.2, cutoff = True, surge = True) # base is threhold used for duration in this case: should try and automate
    wl_bound_surge = tide[:] + surge_da

    # sf.write_forcing() only overwrites boundary points within domain
    sfincs_model.setup_waterlevel_forcing(wl_bound_surge, 'sfincs.bnd')
    sfincs_model.write_forcing() 

    sfincs_model.forcing.pop("precip", None)  # reset
    artificial_precip, start_p, end_p = use.artificial_driver(time_ind, mag = p_mag, dur = int(p_dur), center = center,
                                                                 name = r'Precipitation [$mm$ $h^{-1}$]', lag = p_lag)
    
    sfincs_model.setup_precip_forcing(artificial_precip.to_dataframe())
    sfincs_model.write_forcing() 

    sfincs_model.forcing.pop("dis", None)  # reset
    artificial_discharge, start_d, end_d = use.artificial_driver(time_ind, mag = 0, dur = 0, center = center,
                                                                    name = 1, lag = 0, base = base_flow)
    
    dis_gdf = utils.read_xy(os.path.join(template_folder, "sfincs.src"), crs=sfincs_model.crs)
    sfincs_model.setup_discharge_forcing(timeseries = artificial_discharge.to_dataframe(),
                                         locations = dis_gdf)
    
    
    sfincs_model.write_forcing() 

    real_start =  min(start_s, start_p, start_d).astype('datetime64[s]').astype('datetime64[us]').astype('O')
    real_stop =  max(end_s, end_p, end_d).astype('datetime64[s]').astype('datetime64[us]').astype('O')


    sfincs_model.setup_config(
        tref = "20200101 000000",
        tstart = real_start.strftime('%Y%m%d %H%M%S'),
        tstop = real_stop.strftime('%Y%m%d %H%M%S'),
    )
    sfincs_model.write()

    if plot:
        use.plot_drivers(index, surge_da, wl_bound_surge, artificial_precip, artificial_discharge)
    
    sf_start_t = time.time()

    sfincs_exe = Path("Executables", "SFINCS_executable", "sfincs.exe")
    print('Running SFINCS')
    run_sfincs(
        sfincs_model.root, # path to the SFINCS model root folder
        sfincs_exe=sfincs_exe, # path to the sfincs executable if you want to run SFINCS on windows
        vm=None, # you can use 'docker' if you want to run SFINCS on linux or mac, but you need to install docker desktop first
        verbose = False,
    )

    sf_end_t = time.time()
    sfincs_time = sf_end_t - sf_start_t
    
    # sfincs_model.read_results()

    source_folder = "Models/fiat_model"
    use.duplicate_folder(source_folder, fiat_folder_path)

    # use template to create new hazard map
    ds_hazard = xr.open_dataset('Models/fiat_model/hazard/hazard_map.nc') # change to source_folder later
    max_wl = xr.open_dataset(sfincs_folder_path + '/sfincs_map.nc')['zsmax'][0].transpose('n', 'm').data
    
    ds_hazard['hazard_map'].data = np.nan_to_num(max_wl, nan = 0) * meter_feet # make nan zeros and convert to feet
    hazard_name = 'hazard_map.nc'
    ds_hazard.to_netcdf(fiat_folder_path + '/hazard/' + hazard_name)

    settings = fiat_folder_path + '/settings.toml'
    print('Running Fiat')

    fiat_start_t = time.time()

    cfg = fiat.ConfigReader(settings)
    fiat_model = fiat.FIAT(cfg)
    fiat_model.run()

    fiat_end_t = time.time()
    fiat_time = fiat_end_t - fiat_start_t

    t_dam, prefix = get_damages(Path(fiat_folder_path + "/output", "output.csv"), sub_county=sub_county)
    print(prefix + f' Damage = {t_dam/1e9:.3f} Billion USD')

    # delete folders:
    use.delete_folders(fiat_folder_path + "/hazard")
    use.delete_folders(fiat_folder_path + "/exposure")
    use.delete_folders(fiat_folder_path + "/vulnerability")

    # only keep output.csv
    use.delete_files(fiat_folder_path + "/output", "output.csv")

    ## delete files:
    use.delete_files(sfincs_folder_path, 'sfincs_map.nc')

    # Add gitignores to reduce amount of data on git
    use.add_gitignore(sfincs_folder_path + '/gis', ['*.geojson', '*.tif', '!.gitignore'])
    use.add_gitignore(sfincs_folder_path, ['*.log', '*.txt', '!.gitignore'])
    use.add_gitignore(fiat_folder_path, ['*.log', '*.toml', '!.gitignore'])

    times = np.array([sfincs_time, fiat_time])

    return t_dam, times

def get_damages(file_path, sub_county = None):
    '''
    Collects the damages related to complete, classified and sub-county model depending the sub_county variable
    '''
    out = pd.read_csv(file_path, index_col = "Object ID") # faster by 20 second than using fiat.open_csv
    # assumes in Notebook sub folder
    ind_keep = pd.read_csv('fitted_stats/fiat_indkeep.csv', index_col=0).index # created in fiat_median_tide.ipnb
    out = out.loc[ind_keep]
    prefix = 'Total'
    if sub_county is not None:
        work_df = pd.read_csv('fitted_stats/sub_county.csv', index_col=0)
        classi = pd.read_csv('fitted_stats/classified.csv', index_col=0)
        if sub_county in classi.columns.to_list():
            out = out.set_index('Aggregation Label: Census_block_groups').loc[classi[sub_county].dropna().values]
        else:
            out = out.set_index('Aggregation Label: Census_block_groups').loc[work_df[sub_county].dropna().values]
        prefix = sub_county
    damages = out['Total Damage'].sum()
    t_dam = damages.item()
    return t_dam, prefix

def run_sims_with(index, index_b, folder_to_save, template_folder, sfincs_model,
                  surge_forcing, precip_forcing, base_q = 200, plot = True):
    '''
    Assumes forcings are dataframes with a single column each
    '''
    sfincs_folder_path = "Models/" + folder_to_save + "/sfincs_folder/sfincs_" + str(index) + '_' + str(index_b)
    fiat_folder_path = "Models/" + folder_to_save + "/fiat_folder/fiat_" + str(index) + '_' + str(index_b)

    meter_feet = 3.28084
    # Create new parent folders 
    if index == 0:
        use.create_empty_folder("Models/" + folder_to_save)
        use.create_empty_folder("Models/" + folder_to_save + "/sfincs_folder")
        use.create_empty_folder("Models/" + folder_to_save + "/fiat_folder")
    use.create_empty_folder(sfincs_folder_path)

    model_path = Path(sfincs_folder_path)
    sfincs_model.set_root(model_path, mode = 'w+')
    sfincs_model.write()

    sfincs_model.setup_config(
        tref = surge_forcing.index[0].strftime('%Y%m%d %H%M%S'),
        tstart = surge_forcing.index[0].strftime('%Y%m%d %H%M%S'),
        tstop = surge_forcing.index[-1].strftime('%Y%m%d %H%M%S'),
    )
    sfincs_model.write()

    gdf = utils.read_xy(os.path.join(template_folder,"sfincs.bnd"), crs=sfincs_model.crs)

    sfincs_model.forcing.pop("bzs", None)  # reset
    sfincs_model.setup_waterlevel_forcing(
        timeseries=surge_forcing,
        locations=gdf,
        merge=True,
    )
    sfincs_model.write_forcing()

    sfincs_model.forcing.pop("precip", None)  # reset  
    sfincs_model.setup_precip_forcing(precip_forcing)
    sfincs_model.write_forcing() 

    sfincs_model.forcing.pop("dis", None)  # reset
    artificial_discharge, _, _ = use.artificial_driver(surge_forcing.index, mag = 0, dur = 0, center = surge_forcing.index[0],
                                         name = 1, lag = 0, base = base_q)
    
    dis_gdf = utils.read_xy(os.path.join(template_folder, "sfincs.src"), crs=sfincs_model.crs)
    sfincs_model.setup_discharge_forcing(timeseries = artificial_discharge.to_dataframe(),
                                         locations = dis_gdf)

    if plot:
        sfincs_model.plot_forcing(fn_out="../forcing.png")
        # use.plot_drivers(index, df_surge, df_precip, artificial_discharge)


    sfincs_exe = Path("Executables", "SFINCS_executable", "sfincs.exe")
    print('Running SFINCS')
    run_sfincs(
        sfincs_model.root, # path to the SFINCS model root folder
        sfincs_exe=sfincs_exe, # path to the sfincs executable if you want to run SFINCS on windows
        vm=None, # you can use 'docker' if you want to run SFINCS on linux or mac, but you need to install docker desktop first
        verbose = False,
    )
    
    sfincs_model.read_results()
    max_wl = sfincs_model.results['zsmax'][0]

    source_folder = "Models/fiat_model"
    use.duplicate_folder(source_folder, fiat_folder_path)

    # use template to create new hazard map
    ds_hazard = xr.open_dataset('Models/fiat_model/hazard/hazard_map.nc') # change to source_folder later
    max_wl = xr.open_dataset(sfincs_folder_path + '/sfincs_map.nc')['zsmax'][0].transpose('n', 'm').data
    
    ds_hazard['hazard_map'].data = np.nan_to_num(max_wl, nan = 0) * meter_feet # make nan zeros and convert to feet
    hazard_name = 'hazard_map.nc'
    ds_hazard.to_netcdf(fiat_folder_path + '/hazard/' + hazard_name)

    settings = fiat_folder_path + '/settings.toml'
    print('Running Fiat')
    cfg = fiat.ConfigReader(settings)
    fiat_model = fiat.FIAT(cfg)
    fiat_model.run()

    # check the output
    out = pd.read_csv(Path(fiat_folder_path + "/output", "output.csv"), index_col = "Object ID") # faster by 20 second than using fiat.open_csv
    damages = out['Total Damage']
    # assumes in Notebook sub folder
    ind_keep = pd.read_csv('fitted_stats/fiat_indkeep.csv', index_col=0).index # created in fiat_median_tide.ipnb
    damages = damages.loc[ind_keep].sum()

    t_dam = damages.item()
    print(f'Total Damage = {t_dam/1e9:.3f} Billion USD')

    ## delete folders:
    use.delete_folders(fiat_folder_path + "/hazard")
    use.delete_folders(fiat_folder_path + "/exposure")
    use.delete_folders(fiat_folder_path + "/vulnerability")

    ## only keep output.csv
    use.delete_files(fiat_folder_path + "/output", "output.csv")

    ## delete files:
    use.delete_files(sfincs_folder_path, 'sfincs_map.nc')

    # Add gitignores to reduce amount of data on git
    use.add_gitignore(sfincs_folder_path + '/gis', ['*.geojson', '*.tif', '!.gitignore'])
    use.add_gitignore(sfincs_folder_path, ['*.log', '*.txt', '!.gitignore'])
    use.add_gitignore(fiat_folder_path, ['*.log', '*.toml', '!.gitignore'])

    return t_dam