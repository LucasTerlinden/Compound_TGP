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
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from hydromt_sfincs import utils

## Script used for general convenience and creating artifical boundary conditions

def find_units(df_single):
    start = df_single.name.find('(')
    end = df_single.name.find(')')
    units = df_single.name[start + 1: end]
    index = [start + 1, end]
    return units, index

def duplicate_folder(source_folder, destination_folder):
    try:
        shutil.copytree(source_folder, destination_folder)
        print(f"Folder '{source_folder}' duplicated to '{destination_folder}' successfully.")
    except FileExistsError:
        print(f"Error: Destination folder '{destination_folder}' already exists.")

def create_empty_folder(folder_path):
    try:
        os.mkdir(folder_path)
        print(f"Empty folder created at '{folder_path}'.")
    except FileExistsError:
        print(f"Error: Folder '{folder_path}' already exists.")

def index_first(da):
    '''
    For convenience, place index in front of time in coordinates
    '''
    coord_keys = list(da.coords.keys())
    if 'index' in coord_keys:
        coord_keys.remove('index')  # Remove 'index' if present
        coord_keys.insert(0, 'index')  # Insert 'index' at the beginning
        print(coord_keys)
        da = da.transpose(*coord_keys[:2])
    return da

def create_da(df_single):
    '''
    Convert DataSeries to a DataArray
    '''
    da = xr.DataArray(df_single, 
        dims={'index': df_single.index.to_numpy(dtype='datetime64')}, 
        coords={'index': df_single.index.to_numpy(dtype='datetime64')}, 
        name = df_single.name)
    da = da.rename({'index': 'time'})
    return da

def create_da_from_np(index, values, name):
    df_np = pd.DataFrame({'Time': index, name: values})
    df_np = df_np.set_index('Time')
    da = create_da(df_np[name])
    return da

def artificial_driver(time_index, mag, dur, center, name, lag = 0, base = 0, cutoff = False, surge = False):
    '''
    Currently assumes a gaussian shape

    Parameters
    ----------
    time_index: arr with datetime dtype
    mag: float
    dur: float
        duration in hours of driver
    center: datetime
        datetime of reference location (currently high tide)
    name: str
        name of artificial driver?
    lag: float
        Time lag in hours between driver and reference location
    base: float
        Constant value added to timeseries

    Returns
    ---------
    artificial_da: xr.dataArray
        dataArray containing driver and 

    '''
    resolution = np.diff(time_index)[0] # what is the spacing of the time index
    center = center + np.timedelta64(int(lag), 'h') # if lag, displace center
    total_points = int((time_index[-1] - time_index[0])/(resolution))

    if isinstance(dur, list):
        skew_left = dur[0]
        skew_right = dur[1]
        dur = skew_left + skew_right
        start_left = np.timedelta64(skew_left, 'h')
        end_right = np.timedelta64(skew_right, 'h')
    else:
        skew_left = None
        start_left = np.timedelta64(dur, 'h') / 2
        end_right = np.timedelta64(dur, 'h') / 2

    # identify where in array gaussian will act
    center_ind = int(np.where(time_index == center)[0])
    start = center - start_left
    start_ind = int(np.where(time_index == start)[0])
    end = center + end_right
    end_ind = int(np.where(time_index == end)[0])
    normal_array = np.ones(total_points + 1)
    if cutoff:
        normal_array = normal_array * 0
        if surge and dur < 18:
            base = mag
            skew_left = None
    else:
        normal_array = normal_array * base
    
    if dur != 0:
        if skew_left is None:
            gaussian = define_gaussian(dur, mag, base, start_ind, end_ind)
            normal_array[start_ind:end_ind + 1] = gaussian # zero where surge is not present
        else:
            gaussian_left = define_gaussian(skew_left*2, mag, base, start_ind, center_ind, 2)
            gaussian_right = define_gaussian(skew_right*2, mag, base, center_ind, end_ind, 2)
            normal_array[start_ind:center_ind] = gaussian_left[:len(gaussian_left)//2]
            normal_array[center_ind:end_ind + 1] = gaussian_right[len(gaussian_right)//2:]

    artificial_da = create_da_from_np(time_index, normal_array, name)
    return artificial_da, start, end

def define_gaussian(dur, mag, base, st_ind, end_ind, fudge=1):
    gaussian_space = np.arange(0, fudge * (end_ind - st_ind) + 1, 1)
    # Calculate the Gaussian function
    sigma = dur  # edges are close to zero (common heuristics?)
    mean = np.mean(gaussian_space)
    time_difference = gaussian_space - mean
    gaussian = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((time_difference)/sigma)**2)         
    # ensure gaussian has correct magnitude
    scale = np.max(gaussian)
    gaussian = (mag - base)/scale * gaussian + base
    return gaussian


def plot_drivers(index, surge, wl, precipitation, discharge):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Boundary Conditions for sample {index}')

    wl.plot(ax = axes[0, 0])
    axes[0, 0].set_title('Downstream BC')
    axes[0, 0].set_ylabel(r'Downstream Water Level [$m$]')
    axes[0, 0].set_xlabel('')

    surge.plot(ax = axes[1, 0])
    axes[1, 0].set_title('Surge BC')
    axes[1, 0].set_ylabel(r'Skew Surge [$m$]')

    precipitation.plot(ax = axes[0, 1])
    axes[0, 1].set_title('Precipitation BC')
    axes[0, 1].set_ylabel(r'Magnitude [$mm$ $hr^{-1}$]')
    axes[0, 1].set_xlabel('')

    discharge.plot(ax = axes[1, 1])
    axes[1, 1].set_title('Discharge BC')
    axes[1, 1].set_ylabel(r'Discharge [$m^3$ $s^{-1}$]')
    plt.show()
    return None

def delete_folders(folder_path):
    '''
    Delete folders based on folder path
    '''
    shutil.rmtree(folder_path)
    return None

def delete_files(folder_path, ignore):
    '''
    Delete files in a given folder path unless they
    are specified in the "ignore" param

    Need to find a more efficient way to delete files within folders
    (i.e. if statements are not the best)
    '''
    files = os.listdir(folder_path)
    for file in files:
        if file == 'gis':
            continue
        elif file == 'hydromt.log':
            continue
        elif file == 'sfincs_log.txt':
            continue
        elif file == 'sfincs_his.nc':
            continue
        elif file == ignore:
            continue
        else:
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
    return None

def add_gitignore(folder_name, ignore_patterns):

    # Create the .gitignore file and write ignore patterns to it
    gitignore_path = os.path.join(folder_name, '.gitignore')
    with open(gitignore_path, 'w') as gitignore_file:
        gitignore_file.write('\n'.join(ignore_patterns))

def generate_tide(folder, sfincs_model, mag = 0.98, tidal_p = 12.42, resolution = 600, span = 7 + 6):
    '''
    Writes tidal boundary condition

    Assuming M2 tidal period. Tidal form factor = 0.2 (from NOAA), can therefore assume semi diurnal tide.
    Since M2 = 6.5 * S2 can assume that M2 component is dominating the tidal period more frequently.
    This assumption is also reinforced by the fact that skew surge can occur at any beating.
    Therefore, a tide could be sampled which represent the lower high tide, a higher high tide or a neap tide.
    If a higher high spring tide is sampled, this assumption will be conservative, as the high tide will be
    constant and not decrease as a spring tide would in nature.

    Could maybe take the tidal cycle which has been observed in the past.

    Parameters
    ----------
    folder: path
        Path to folder in which bnd file can be found
    sfincs_model:
        sfincs model that has been initialized outside of function
    mag: float
        Magnitude of tidal cycle [m]
    tidal_p: float
        Period of the tidal cycle used [hours]
    resolution: float
        time interval of model [sec]
    span: flaot
        Time span of boundary condition [days]
    '''
    t = np.arange(0, 3600/resolution * 24 * span + 1, 1) * resolution
    df_drivers = pd.read_csv('Data/all_drivers.csv')
    df_drivers = df_drivers.set_index(pd.to_datetime(df_drivers.iloc[:, 0]))
    df_drivers = df_drivers.drop(columns = 'DateTime(UTC)')
    mean_sl = df_drivers['WL_trend (m)'].values.max() # take highest observed average sea level
    offset = tidal_p/4 # ensures that model start at high tide
    y = mag*(np.sin(2*np.pi*(t + offset*60*60)/(tidal_p*60*60))) + mean_sl

    df_tide = pd.DataFrame({'time': t, 1: y})
    df_tide = df_tide.set_index('time')
    gdf = utils.read_xy(os.path.join(folder,"sfincs.bnd"), crs=sfincs_model.crs)

    sfincs_model.forcing.pop("bzs", None)  # reset
    sfincs_model.setup_waterlevel_forcing(
        timeseries=df_tide,
        locations=gdf,
        merge=True,
    )
    sfincs_model.write_forcing()
    return None

def retrieve_historical_tide(folder, sfincs_model, mag, span = 7 + 6):
    np.random.seed(5) # daily inequality is ~25 cm, if a magnitude is sampled twice, it needs to be reproducible to not create confusion

    df_drivers = pd.read_csv('Data/all_drivers.csv')
    df_drivers = df_drivers.set_index(pd.to_datetime(df_drivers.iloc[:, 0]))
    df_drivers = df_drivers.drop(columns = 'DateTime(UTC)')

    mean_sl = df_drivers['WL_trend (m)'].values.max() # take highest observed average sea level


    tidal_peaks = pd.read_csv('Data/skew_surge_tides.csv', parse_dates = ['DateTime(UTC)'])
    tidal_peaks.set_index('DateTime(UTC)', inplace = True)
    cond_hh = tidal_peaks[tidal_peaks['Type'] == 'HH']
    ss_hh = cond_hh.iloc[:, [1]].round(2)
    mag = np.round(mag, 2)
    tidal_mag = ss_hh[ss_hh == mag].dropna()
    if len(tidal_mag) > 1:
        random_index = np.random.randint(0, len(tidal_mag))
        print(random_index)
    elif len(tidal_mag) == 1:
        random_index = 0
    else:
        raise Exception('!! Tide does not exist') 
    
    sampled_tide = tidal_mag.index[random_index]
    sampled_tide += np.timedelta64(6, 'h')
    start_time = sampled_tide - np.timedelta64(int(span/2*24), 'h')
    end_time = sampled_tide + np.timedelta64(int(span/2*24), 'h')

    tide_time_series = df_drivers.loc[start_time:end_time, 'Tidal (m)'].resample('10min').interpolate(method='linear')
    t = np.arange(0, span * 24 * 60 * 60 + 600, 600)
    y = tide_time_series.values + mean_sl
    df_tide = pd.DataFrame({'time': t, 1: y})
    df_tide = df_tide.set_index('time')
    gdf = utils.read_xy(os.path.join(folder,"sfincs.bnd"), crs=sfincs_model.crs)

    sfincs_model.forcing.pop("bzs", None)  # reset
    sfincs_model.setup_waterlevel_forcing(
        timeseries=df_tide,
        locations=gdf,
        merge=True,
    )
    sfincs_model.write_forcing()

    peak_ind = len(t)//2
    return peak_ind
