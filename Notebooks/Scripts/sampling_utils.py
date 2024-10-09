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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from hydromt_sfincs import SfincsModel
from matplotlib.ticker import ScalarFormatter
from pdf2image import convert_from_path

import Notebooks.Scripts.normalization as normalizer
import Notebooks.Scripts.Useful as use
from Notebooks.Scripts.run_sfincs_and_fiat import get_damages, run_simulation

# Script used to run the MDA, TGP and plot information related to these

def get_forcing(denorm_df, clock, bonus = None):
    '''
    Recieves denorm_df, which contains the stochastic event set.
    If a variable is not contained, it will use the median of the histroical event set.
    If other constants want to be used. A dataframe in the form of bonus can be provided.
    Currently hardcoded with try and except blocks.

    Returns the values to form the boundary condtions of an event.
    '''
    constants = pd.read_csv('fitted_stats/historical_eventset.csv')
    medians = constants.median()
    if bonus is not None:
        medians[bonus.columns.to_list()] = bonus.values.flatten()

    try:
        surge_mag = denorm_df.loc[clock, 'S Mag [m]']
    except KeyError:
        raise Exception('Surge should always be defined')
    
    try:
        precip_mag = denorm_df.loc[clock, 'P Mag [mm/hr]']
    except KeyError:
        raise Exception('Precipitation should always be defined')
    
    try:
        surge_dur = denorm_df.loc[clock, 'S Dur [tidal cycles]']
    except KeyError:
        surge_dur = medians['S Dur [tidal cycles]']

    try:
        precip_dur = denorm_df.loc[clock, 'P Dur [hr]']
    except KeyError:
        precip_dur = medians['P Dur [hr]']

    try:
        precip_lag = denorm_df.loc[clock, 'P Lag [hr]']
    except KeyError:
        precip_lag = medians['P Lag [hr]']

    try:
        tide_mag = denorm_df.loc[clock, 'T Mag [m]']
    except KeyError:
        tide_dist = pd.read_csv('fitted_stats/emperical_tide.csv', header = None) # take median higher high tide
        tide_mag = tide_dist.median().item()

    try:
        base_q = denorm_df.loc[clock, 'Q Mag [m^3/hr]']
    except KeyError:
        base_q = 200
    return surge_mag, precip_mag, surge_dur, precip_dur, precip_lag, tide_mag, base_q

def run_MDA(model_folder, denorm_subset, num_sims, output = ['Total'], bonus = None,
            plot = True, artificial_tide = False):
    '''
    Loops through the files of denorm_subset to obtain the damages and times associated with each event.
    This is done by running the coupled SFINCS and FIAT models.
    Often used to obtain the damages associated with an MDA subset. It is also used to obtain all the 
    damages of the artifical floods in the training event set.

    Parameters
    ---------
    model_folder: str
        Path at which SFINCS, FIAT, damages and times will be saved
    stoc_set: Pandas.DataFrame
        Contains the complete stochastic event set for an arbitrary number of dimensions
        in the real space.
    num_sims: int
        Total number of artifical floods that can be simulated (len("stoc_set") often used)
    output: lst of strings
         Default: ['Total']. Name of outputs for which the economic damage needs to be known
    bonus: pandas.DataFrame
        No default. Use if number of dimensions is smaller than 6, and a different constant value
        for a probablistic variable is desired. Currently uses the median of the stochastic event set.
        Requires at least the correct column names of probabilistic variables to change and a row with 
        corresponding values
    plot: bool
        Default: True. Plot the boundary conditions of the artifical flood event.
    artifical_tide: bool
        Default: False. If true will schematize the tide as a constant harmonic which has
        the same period as the M2 tidal signal. If false, will sample a historical higher high
        tidal signal which has the same tidal magnitude as artificial flood event.

    Output
    -------
    target: pandas.DataFrame of size num_sims x len(output)
        The economic damages associated with the artifical floods in the subset and the outputs
        in "output"
    times: pandas.DataFrame of size num_sims x 2
        The computational time of SFINCS and FIAT for each artifical flood in the subset
    save_paths: lst of strings
        Contains the path of the target and times output which were saved to seperate csv files.
    '''
    save_path = 'Models/' + model_folder + '/damages.csv'
    save_path_t = 'Models/' + model_folder + '/times.csv'
    save_paths = [save_path, save_path_t] # third output of function

    # Deterministic approach: if function is stopped for whatever reason, it can read the 
    # the files saved to continue where it stopped.
    if os.path.exists(save_path):
        target = pd.read_csv(save_path, index_col=0)
        clock = len(target[target[output[0]] != 0])
        times = pd.read_csv(save_path_t, index_col=0)
    # If this is the first time running this function, initialize first and second outputs
    else:
        clock = 0
        target = pd.DataFrame(np.zeros((num_sims, len(output))), columns = output)
        times = pd.DataFrame(np.zeros((num_sims, 2)), columns = ['SFINCS', 'FIAT'])
    
    # loop through artifical floods in subset and run the coupled SFINCS and FIAT models for each        
    for i in range(len(denorm_subset) - clock):
        # Initialize SFINCS model
        template_folder = Path('Models/FloodAdapt_stolen')
        sf = SfincsModel(root=template_folder, mode='r')
        sf.read()
        # obtain paramaters for boundary condition schematization
        surge_mag, precip_mag, surge_dur, precip_dur, precip_lag, tide_mag, base_q = get_forcing(denorm_subset, clock, bonus = bonus)
        # short cut to obtain damages for multiple outputs. First ask run_simulation function
        # for damages associated with first output. Then, loop through the remaining outputs
        # on the csv created by the FIAT model.
        if len(output) == 1:
            sub_county = None
        else:
            sub_county = output[0]
        # obtain the damages and times for the artifical flood that is being numerically simulated
        target[output[0]].iloc[clock], time_i = run_simulation(clock, model_folder, template_folder, sf, surge_mag, surge_dur, precip_mag,
                                                       precip_dur, tide_mag = tide_mag, p_lag = precip_lag, base_flow = base_q,
                                                       plot = plot, sub_county = sub_county, artificial_tide=artificial_tide) # assume constant discharge
        
        # retrieve the file path of the csv created by the FIAT model
        fiat_folder_path = "Models/" + model_folder + "/fiat_folder/fiat_" + str(clock)
        if sub_county is not None:
            # loop through all outputs to obtain the damages associated with these
            for z in range(len(output) - 1):
                t_dam, _ = get_damages(Path(fiat_folder_path + "/output", "output.csv"), sub_county = output[z + 1])
                target[output[z + 1]].iloc[clock] = t_dam

        # Incrementally save outputs in case function needs to be stopped
        times.iloc[clock, :] = time_i
        target.to_csv(save_path)
        times.to_csv(save_path_t)
        clock += 1
    return target, times, save_paths


def run_tgp(folder_csv, model_folder, num_samples, denorm_subset,
            minmax_scaler, ex_rate, prior = 'bflat', output = ['Total'],
            num_sims = 10_000, tgp = True, MDA_ran = True, bonus = None, artificial_tide = False):
    '''
    Run the (T)GP. Assumes the sampled, available and min max are alreadly located in csvs.
    
    Parameters
    ------
    folder_csv: lst str
        contains the list of folder paths where statistics related to different outputs will be saved
    model_folder: str
        name of the folder in the Models parent folder where the SFINCS and FIAT simulations will be saved
    num_samples: int
        maximum number of samples before stopping
    denorm_subset: pd.DataFrame
        subset containing the events that have been used for the MDA
    minmax_scaler: sklearn scaler object
        scaler to normalize/denormalize stochastic event set
    ex_rate: float
        extreme rate of the stochastic event set
    prior: str
        Default: bflat. Default used by the TGP in R.
        Prior used by the Treed Gaussian Process to solve hierarchical equations.
    output: lst of strings
         Default: ['Total']. Name of outputs for which the economic damage needs to be known
    num_sims: int
        Total number of artifical floods that can be simulated (len("stoc_set") often used)
    tgp: bool
        If True, runs a TGP. If False, runs a GP
    MDA_ran: bool
        Default: True. Set to false if the (T)GP needs to be run without an MDA
    bonus: pandas.DataFrame
        No default. Use if number of dimensions is smaller than 6, and a different constant value
        for a probablistic variable is desired. Currently uses the median of the stochastic event set.
        Requires at least the correct column names of probabilistic variables to change and a row with 
        corresponding values
    artificial_tide: bool
        Default: False. If true will schematize the tide as a constant harmonic which has
        the same period as the M2 tidal signal. If false, will sample a historical higher high
        tidal signal which has the same tidal magnitude as artificial flood event.
    '''
    
    full_break = False # when true, the TGP will stop sampling
    # easier to use an array than a list when looping through outputs to obtain their damages
    # for an artifical flood
    base_output = np.array(output)
    # easier to use an array than a list when deleting indices (stop crit reached for output)
    mod_output = np.array(output)
    # create numerical indices for each output
    output_index = np.arange(1, len(output) + 1)

    # create empty folders for each output
    for i in range(len(folder_csv)):
        use.create_empty_folder(folder_csv[i])

    # Obtain the number of times the TGP needs to be fitted before the same output 
    # is found again in the round robbin schedule.
    if MDA_ran:
        crit_read = len(denorm_subset) + len(output)
    else:
        crit_read = len(output) - 1

    save_path = 'Models/' + model_folder + '/damages.csv'
    save_path_t = 'Models/' + model_folder + '/times.csv'
    # If save paths already exist (i.e. created with the MDA), obtain the damages
    # for the given output and the SFINCS and FIAT times.
    if os.path.exists(save_path):
        target = pd.read_csv(save_path, index_col=0)
        clock = len(target[target[output[0]] != 0])
        times = pd.read_csv(save_path_t, index_col=0)
    else: # initalize damages and times if these do not already exist
        clock = 0
        target = pd.DataFrame(np.zeros((num_sims, len(output))), columns = output)
        times = pd.DataFrame(np.zeros((num_sims, 2)), columns = ['SFINCS', 'FIAT'])

    # initialize a memory bank to know when stop crit is reached for:
    # (a) a particular output
    # (b) all outputs
    bool_lst = len(output) * [False]
    bool_dict = dict(zip(output, bool_lst)) # (b)

    alm_crit = 0.1 # stopping criterion threshold, hardcode
    # stopping criterion based on 2 consecutive values below a threshold (alm_crit)
    memory = pd.DataFrame(np.ones((2, len(output))), columns = output) # (a)

    # For a given output, read previous metrics or initialize them (if first time)
    for i in range(num_samples - clock):
        for out_var in mod_output: # loop through outputs which have not yet reached stop crit
            # obtain an array of booleans where only the current output is set to True
            bool_ind = base_output == out_var
            ind_out = output_index[bool_ind].item() # obtain index corresponding to output
            # For a given output, read previous metrics (i) or initialize them (if first time) (ii)
            if clock >= crit_read: # (i)
                df_stats = pd.read_csv(folder_csv[ind_out] + 'stability.csv', index_col = 0)
                alm_stats = df_stats.loc[:, 'Mean. Unc'].tolist()
                ead_list = df_stats.loc[:, 'EAD_mean'].tolist()
                ead5_list = df_stats.loc[:, 'EAD_5'].tolist()
                ead95_list = df_stats.loc[:, 'EAD_95'].tolist()
                time_list = df_stats.loc[:, 'Time [s]'].tolist()
                re_val_list = df_stats.iloc[:, -1].tolist()
            else: # (ii)
                alm_stats = []
                ead_list = []
                ead5_list = []
                ead95_list = []
                time_list = []
                re_val_list = []

            # According to bool of TGP choose which executable to use.
            if tgp:
                r_exec = 'Notebooks/Scripts/R_scripts/tgp_exec.R'
            else:
                r_exec = 'Notebooks/Scripts/R_scripts/gp_exec.R'

            samples = 1 # number of artifical floods to pick from TGP iteration (hardcoded)
            # Prepare inputs/outputs for the TGP in R
            # folder_csv index 0 represents the parent folder, not the first output
            ## inputs
            sol_file = folder_csv[0] + 'sampled_events.csv' # sampled
            opt_file = folder_csv[0] + 'sample_space.csv' # available

            ## outputs part 1
            acqui_file = folder_csv[ind_out] + 'acqui.csv' # ALM stat for available
            X_file = folder_csv[ind_out] + 'X_mean.csv' # sampled mean
            XX_file = folder_csv[ind_out] + 'XX_mean.csv' # available mean
            plot_file = folder_csv[ind_out] + 'Plot.pdf' # if 2d, plot of current TGP

            ## outputs part 2
            X5_file = folder_csv[ind_out] + 'X_5.csv' # sampled 5% quantile
            X95_file = folder_csv[ind_out] + 'X_95.csv' # sampled 95% quantile
            XX5_file = folder_csv[ind_out] + 'XX_5.csv' # available 5% quantile
            XX95_file = folder_csv[ind_out] + 'XX_95.csv' # available 95% quantile

            run_r_exec = ('Rscript ' + r_exec + ' ' + sol_file + ' ' + opt_file + ' ' + acqui_file + ' ' + 
                        X_file + ' ' + XX_file + ' ' + plot_file + ' ' + out_var + ' ' + X5_file + ' ' + 
                        X95_file + ' ' + XX5_file + ' ' + XX95_file + ' ' + prior)

            # read inputs from parent folder, saved before using the function
            df_sampled = pd.read_csv(folder_csv[0] + 'sampled_events.csv', index_col = 0)
            df_available = pd.read_csv(folder_csv[0] + 'sample_space.csv', index_col = 0)
            df_min_max = pd.read_csv(folder_csv[0] + 'max_min.csv', index_col = 0)
            # hardcode scaler used (min max scaler)
            max = df_min_max.iloc[:,0].values.flatten()
            min = df_min_max.iloc[:,1].values.flatten()


            start_time = time.time()
            subprocess.run(run_r_exec, shell=True, capture_output=True, text=True) # run TGP
            end_time = time.time()
            execute_time = end_time - start_time # compute TGP time
            time_list.append(execute_time)

            # if 2 dimensional, save plots to a dedicated folder called "ALM"
            if denorm_subset.shape[1] == 2: # i.e. can still plot
                folder_figs = folder_csv[ind_out] + 'ALM/'
                use.create_empty_folder(folder_figs)
                images = convert_from_path(plot_file)
                save_path_img = folder_figs + 'plot' + str(len(df_sampled)) + '.png'
                images[0].save(save_path_img)

            # model the risk associated with TGP surrogate model
            ead_mean, mean_risk = obtain_cons_curve(X_file, XX_file, min[ind_out - 1], max[ind_out - 1], ex_rate)
            ead_list.append(ead_mean) 

            ead_5, five_risk = obtain_cons_curve(X5_file, XX5_file, min[ind_out - 1], max[ind_out - 1], ex_rate)
            ead5_list.append(ead_5)

            ead_95, ninefive_risk = obtain_cons_curve(X95_file, XX95_file, min[ind_out - 1], max[ind_out - 1], ex_rate)
            ead95_list.append(ead_95)

            # read alm metrics for available samples, largest value(s) = next artifical flood(s) to simulate
            df_alm = pd.read_csv(acqui_file)
            df_alm = df_alm.sort_values(by = 'x')
            indicies = df_alm.index[-samples:]
            # alm_imp = df_alm.iloc[-samples:].values
            mean_alm = df_alm.mean().item()
            alm_stats.append(mean_alm)

            # assess the 100 year return period and monitor its modeling uncertainty
            re_per = 100
            risk_pd_list = [five_risk, mean_risk, ninefive_risk]
            re_val_list_i = []
            for i in range(len(risk_pd_list)):
                df = risk_pd_list[i]
                df['Difference'] = abs(df['Prob'] - 1/re_per)
                closest_row_index = df['Difference'].idxmin() # obtain the closest Retrun period

                closest_re_per = df.at[closest_row_index, 'Conseq']
                re_val_list_i.append(closest_re_per)
            
            re_val_list.append(re_val_list_i)

            # save metrics
            df_stats = pd.DataFrame({'EAD_5': ead5_list,
                                    'EAD_mean': ead_list,
                                    'EAD_95': ead95_list,
                                    'Mean. Unc': alm_stats,
                                    'Time [s]': time_list,
                                    str(re_per) + ' RP': re_val_list})#, index = np.arange(len(subset), clock + 1))
            df_stats.to_csv(folder_csv[ind_out] + 'stability.csv')

            if clock == (num_samples + 1): # clock is always one ahead if above sfincs-fiat loop
                print('Breaking...\n'
                    'Number of samples reached')
                full_break = True
                break
            else:
                None
            
            print(f"Currently have {clock} samples")

            # assess stop criterion for particular output
            output_memory = memory[out_var]
            bool = output_memory != 1.0 # create a x2 boolean
            stop_ind = len(output_memory[bool]) # index first false in boolean (i.e. 0 or 1)
            if mean_alm < alm_crit:
                memory.loc[stop_ind, out_var] = mean_alm # provide a non 1 value (1 associated with False)
            else:
                memory.loc[:, out_var] = 1.0 # ensure that the values are consecutive

            if all(memory[out_var].values < alm_crit): # if both values for output are smaller than stop crit
                print('No longer need to sample from: ' + out_var)
                bool_dict[out_var] = True
                mod_output = mod_output[mod_output != out_var] # no longer sample from this dimension

                # save the samples needed to fit this specific output to output folder
                df_sampled.to_csv(folder_csv[ind_out] + 'sampled_events.csv')
                df_available.to_csv(folder_csv[ind_out] + 'sample_space.csv')
                df_min_max.iloc[ind_out - 1].to_csv(folder_csv[ind_out] + 'max_min.csv')

            if all(bool_dict.values()): # all outputs have reached stop crit
                print('Stopping Criterion Met!')
                full_break = True
                break
            
            # retrieve artifical flood with highest ALM and denormalize to the real space
            gp_samples = df_available.iloc[indicies]
            denorm_samples = normalizer.denormalize_dataset(gp_samples, minmax_scaler).values
            denorm_samples = pd.DataFrame(denorm_samples, columns = denorm_subset.columns)
                
            for j in range(samples):
                # intialize SFINCS model
                template_folder = Path('Models/FloodAdapt_stolen')
                sf = SfincsModel(root=template_folder, mode='r')
                sf.read()
                # obtain paramaters for boundary condition schematization
                surge_mag, precip_mag, surge_dur, precip_dur, precip_lag, tide_mag, base_q = get_forcing(denorm_samples, j, bonus = bonus)
                # short cut to obtain damages for multiple outputs. First ask run_simulation function
                # for damages associated with current output. Then, loop through the remaining outputs
                # on the csv created by the FIAT model.
                if out_var == 'Total':
                    sub_county = None
                else:
                    sub_county = out_var
                # obtain damages for current output, and time for SFINCS and FIAT
                target[out_var].iloc[clock], times.iloc[clock, :] = run_simulation(clock, model_folder, template_folder, sf, surge_mag, surge_dur, precip_mag,
                                            precip_dur, tide_mag = tide_mag, p_lag = precip_lag, base_flow = base_q, sub_county = sub_county, artificial_tide=artificial_tide) # assume constant discharge
                fiat_folder_path = "Models/" + model_folder + "/fiat_folder/fiat_" + str(clock)
                if sub_county is not None:
                    # loop through other outputs to obtain their damages
                    for z in range(len(output)):
                        if output[z] != sub_county:
                            t_dam, _ = get_damages(Path(fiat_folder_path + "/output", "output.csv"), sub_county = output[z])
                            target[output[z]].iloc[clock] = t_dam
                        else:
                            None
                # Incrementally save in case function needs to be stopped
                target.to_csv(save_path)
                times.to_csv(save_path_t)

                # normalize the target(s) obtained, and add to the current subset
                gp_consequences = (target.iloc[clock].values - min) / (max - min)
                new_row = np.concatenate((gp_samples.values[j], gp_consequences), axis = 0).reshape(1, -1)
                new_row_df = pd.DataFrame(new_row, columns = df_sampled.columns)
                df_sampled = pd.concat([df_sampled, new_row_df], ignore_index=True)
                clock += 1

            # Ensure min max scaler is still between 0 and 1 with new artificial flood(s)
            norm_cons = df_sampled.iloc[:, -len(output):].values
            re_norm_cons, min, max = normalizer.change_norm(norm_cons, min, max)
            df_sampled.iloc[:, -len(output):] = re_norm_cons
            scaler_cons = pd.DataFrame({'Max Cons': max, 'Min Cons': min})
            scaler_cons.to_csv(folder_csv[0] + 'max_min.csv') # save

            # remove artificial floods that have been simulated
            df_available = df_available.drop(indicies).reset_index(drop=True)

            ## save current input of TGP to parent folder
            df_sampled.to_csv(folder_csv[0] + 'sampled_events.csv')
            df_available.to_csv(folder_csv[0] + 'sample_space.csv')
            ind_out += 1
        if full_break:
            break
    return None

def get_return_periods(x, a=0.0, extremes_rate=1.0):
    assert np.all(np.isfinite(x))
    b = 1.0 - 2.0 * a
    ranks = (len(x)+1) - scipy.stats.rankdata(x, method="average")
    freq = ((ranks - a) / (len(x) + b)) * extremes_rate
    rps = 1 / freq
    return rps

def obtain_cons(cons_scaled, min = None, max = None, ex_rate = 1.0):
    consequences = cons_scaled
    if min is not None:
        denorm_c = consequences * (max - min) + min # denormalize
    else:
        denorm_c = consequences
    np.random.seed(5)
    random_num = np.random.rand(cons_scaled.shape[0])
    denorm_c = np.maximum(random_num, denorm_c.flatten()) # ensure consequences are always positive
    if np.isnan(denorm_c).any():
        ead = np.nan
        df_risk = None
    else:
        obs_rps = get_return_periods(denorm_c, extremes_rate = ex_rate)
        df_risk = pd.DataFrame({'Conseq': denorm_c.flatten(),
                                   'Prob': 1/obs_rps})
        df_risk = df_risk.sort_values('Prob')
        ead = scipy.integrate.trapezoid(y = df_risk.iloc[:, 0].values, x = df_risk.iloc[:, 1].values)
    return ead, df_risk

def obtain_cons_curve(sampled_file, available_file, min, max, extreme_rate):
    df_X_means = pd.read_csv(sampled_file)
    df_XX_means = pd.read_csv(available_file)

    consequences_gp = np.concatenate((df_X_means.values, df_XX_means.values))
    ead_gp, df_risk_gp = obtain_cons(consequences_gp, min, max, extreme_rate)
    return ead_gp, df_risk_gp

def obtain_ead_interp(df_interp, min, max, extreme_rate):
    consequences = df_interp.iloc[:, -1].values
    ead, df_risk = obtain_cons(consequences, min, max, extreme_rate)
    return ead, df_risk

def color_dic(num_dim):
    color_dic = {2: 'b',
                 3: '#ff7f0e',
                 4: 'r',
                 5: 'c',
                 6: 'k'}

    c = color_dic[num_dim]
    return c

def plot_ead(df_stats, dimension):
    num_dim = dimension
    label = str(num_dim) + ' dims'
    c = color_dic(num_dim)
    ax_stats = df_stats.plot(y = 'EAD_mean', c = c, label = f'Mean for ' + label)
    ax_stats.set_xlabel('Number Of TGP Iterations [-]', fontsize = 12)
    ax_stats.set_ylabel('EAD [USD]', fontsize = 12)
    ax_stats.set_ylim([0, 8e8])
    label_fill = 'TGP Uncertainty'

    x = df_stats.index
    y1 = df_stats.loc[:, 'EAD_5']
    y2 = df_stats.loc[:, 'EAD_95']

    ax_stats.fill_between(x, y1, y2, color = c, alpha=0.3, label = label_fill)
    ax_stats.grid()
    ax_stats.legend(fontsize = 12)
    return ax_stats

def plot_unc(samples, uncertainty):
    plt.plot(samples, uncertainty)
    plt.grid()
    plt.xlabel('Samples used in Gaussian Process')
    plt.ylabel('Predictive Quantile Difference')
    plt.title('Largest Uncertaintiy of Available Samples at Each Iteration')
    return None

def plot_fits(consequences_df, EAD, dimensions, ax_cons = None, cons_five = None, cons_95 = None,
              num_dim = None, label = None, color = None):
    if label is not None:
        label = label
        c = color

    elif num_dim is None:
        num_dim = len(dimensions)
        label = str(num_dim) + ' dimensions'
        c = color_dic(num_dim)
    else:
        num_dim = num_dim + 2
        list_storage = ['b0', 'bflat', 'bmle']
        label = list_storage[num_dim - 2] + ' beta prior'
        c = color_dic(num_dim)
    consequences_df['Return Period'] = 1 / consequences_df.iloc[:, 1]
    if ax_cons is not None:
        consequences_df.plot(y = 'Conseq', x = 'Return Period', c = c, s = 0.1, kind = 'scatter',
                             label = f'Samples for ' + label + f' (EAD = {EAD/1e6:.2f} Mil. USD)',
                             ax = ax_cons)
    else:
        ax_cons = consequences_df.plot(y = 'Conseq', x = 'Return Period', c = c, s = 0.1, kind = 'scatter',
                                       label = f'Samples for ' + label + f' (EAD = {EAD/1e6:.2f} Mil. USD)',
                                       figsize = (10, 10))
        # ax_cons.set_title(f'Consequences for {len(consequences_df)} events')
        ax_cons.set_xscale('log')
        ax_cons.xaxis.set_major_formatter(ScalarFormatter())
    ax_cons.grid()
    ax_cons.set_ylabel('Return value [USD]')
    ax_cons.set_xlabel('Return period [year]')
    if cons_five is not None: # assumes both are knwon then
        cons_five['Return Period'] = 1 / cons_five.iloc[:, 1]
        cons_95['Return Period'] = 1 / cons_95.iloc[:, 1]
        ax_cons.fill_between(cons_five['Return Period'], cons_five.iloc[:, 0],
                             cons_95.iloc[:, 0], color=c, alpha=0.3, label = 'TGP-LLM uncertainty for ' + label)
    ax_cons.legend()
    return ax_cons


def plot_sample_loc(df_sampled, df_available, num_mda):
    '''
    For scaled variables
    '''
    values = df_sampled.iloc[num_mda:, :-1].values
    subset_mda = df_sampled.iloc[:num_mda, :-1].values
    available = df_available.values
    num_dimensions = values.shape[1]
    num_values = values.shape[0]

    fig, axes = plt.subplots(1, num_dimensions, figsize=(num_dimensions*2.5, 8), sharey=True)

    for dim, ax in enumerate(axes, start=1):
        x_coordinates = np.full(num_values, 0.3) 
        ax.scatter(x_coordinates, values[:, dim-1], color='blue', s = 5, alpha=1, label = 'TGP LLM')
        x_coordinates = np.full(num_mda, 0.3) 
        ax.scatter(x_coordinates, subset_mda[:, dim-1], color = 'red', s = 5, alpha=1, label = 'MDA')
        x_coordinates = np.full(len(available), 0.7)
        ax.scatter(x_coordinates, available[:, dim-1], color='black', marker = 'x', s = 5, alpha=1, label = 'Remaining')
        ax.set_xlim(0, 1)
        # Remove x-axis ticks and labels
        ax.xaxis.set_ticks([])
        ax.set_title(df_sampled.columns[dim - 1])
        ax.tick_params(axis='y', rotation=90)
        ax.set_aspect(5)

    axes[0].set_ylabel('Min Max Values')
    plt.suptitle('Sampled Values')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1))
    return None
