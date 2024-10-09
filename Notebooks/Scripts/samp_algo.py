# Compound_MLA
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

import subprocess
import time

import numpy as np
import pandas as pd
from pdf2image import convert_from_path

import Notebooks.Scripts.normalization as normalizer
import Notebooks.Scripts.sampling_utils as sam_util
import Notebooks.Scripts.Useful as use
from Notebooks.Scripts import selector_mda


def mda_tgp(stoc_set, model_folder, cur_folder, output = ['Total'], total_samples = 100,
            restart = True, seed = None, mda_runs = None, bonus = None, tgp = True, artificial_tide = False,
            mda_plot = False):
    '''
    Functions runs a coupled Maximum Dissimilarity Algorithm (MDA) and Treed Gaussian Process (TGP)
    sampling algroithm. MDA runs for a pre deterimned number of sampled. TGP stops if 
    stability is reached, or if total samples is met.

    Author: Lucas Terlinden-Ruhl

    Parameters
    ---------
    stoc_set: str or Pandas.DataFrame
        Path to event set, or Pandas.DataFrame which contains the event set
        Contains the complete stochastic event set for an arbitrary number of dimensions
        in the real space.
    model_folder: str
        Path at which SFINCS, FIAT, damages and times will be saved
    cur_folder: str
        Path to which statistics created by MDA and TGP will be saved.
    output: lst of strings
        Default: ['Total']. Name of outputs the TGP will sample from in a round robbin fashion.
    total_samples: int
        Default: 100. If the number of simulations exceeds this value, the sampling will stop
    restart: bool
        Default: True. Set to False if you want to continue sampling an existing event set
    seed: int
        Index to initialize the MDA algorithm. Default is maximum of 'S Mag' Parameter
    mda_runs: int
        Number of MDA samples. Defualt is the number of verticies (i.e. number of dimensions squared)
    bonus: pandas.DataFrame
        No default. Use if number of dimensions is smaller than 6, and a different constant value
        for a probablistic variable is desired. Currently uses the median of the stochastic event set.
        Requires at least the correct column names of probabilistic variables to change and a row with 
        corresponding values
    tgp: bool
        True by default. Set to False if a Gaussian Process is wanted instead (faster, but does not partition)
        Will use TGP blackbox in R. 
    artificial_tide: bool
        if False will sample a historical tide with magnitude: t_mag. If True will generate an artifical signal with an M2 tidal period.
    mda_plot: bool
        Plot the boundary conditions of the mda samples. Default is False.

    Output
    -------
    None. Important files are:
        stability.csv: contains EAD, highest uncertainty, time and 100 year RP for each (T)GP model
        minmax.csv: contains the min max values of each output to denormalize outputs.
        sampled_events: contains the sampled events and their normalized output. minmax scaler is not
            saved, but can be easily recreated using the original stochastic event set and the
            normalizer.scaler function.   
    '''
    if type(stoc_set) == str:
        stoc_set = pd.read_csv(stoc_set)

    # expects stoc set to be in the real space
    
    minmax_scaler = normalizer.scaler(stoc_set)
    # stoc set has each dimension scaled between 0 and 1, less bias for MDA and TGP
    df_minmax_scaled = normalizer.normalize_dataset(stoc_set, minmax_scaler)

    # initialization of MDA, default is maximum of surge magnitude (first dimension)
    if seed is None:
        seed = df_minmax_scaled['S Mag'].argmax()

    # size of the subset of the MDA
    if mda_runs is None:
        mda_runs = 2**stoc_set.shape[1]

    # create the MDA subset bassed on the seed, stochastic event set, and size of the subset
    maxmin_class = selector_mda.MaxMin()
    lst_ind = maxmin_class.select_from_cluster(df_minmax_scaled.values, mda_runs, seed)
    subset = df_minmax_scaled.iloc[lst_ind].copy(deep = True)

    # denormalize the MDA subset from minmax to real space
    denorm_subset = normalizer.denormalize_dataset(subset, minmax_scaler).values
    denorm_subset = pd.DataFrame(denorm_subset, columns = stoc_set.columns)
    if restart: # restart entire process, since MDA is deterministic it will not be rerun
        print('Starting MDA')
        # run the coupled sfincs and fiat models for each artifical flood event in MDA subset
        # target: damages associated with flood event for each output
        # times: comp time of SFINCS and FIAT for each artificial flood
        target, times, paths = sam_util.run_MDA(model_folder, denorm_subset, len(stoc_set),
                                            output = output, bonus = bonus, plot = mda_plot,
                                            artificial_tide=artificial_tide)
        target.iloc[mda_runs:] = 0 # ensure restart is satisfied
        times.iloc[mda_runs:] = 0
        target.to_csv(paths[0])
        times.to_csv(paths[1])
        print('MDA Finished')
        
        # normalize the damages of the artifical floods associated with the MDA subset
        target_minamax_scaled, min, max = normalizer.quick_normalizer(target.values[:mda_runs])
        df_sampled = subset.copy()
        # df_sampled are the simulated artifical floods
        df_sampled[output] = target_minamax_scaled
        # df_available are the artifical floods that have not yet been simulated
        df_available = df_minmax_scaled.drop(subset.index).reset_index(drop=True)

        # save the two input files + scaler
        use.create_empty_folder(cur_folder)
        df_sampled.to_csv(cur_folder + 'sampled_events.csv')
        df_available.to_csv(cur_folder + 'sample_space.csv')
        scaler_cons = pd.DataFrame({'Max Cons': max, 'Min Cons': min})
        scaler_cons.to_csv(cur_folder + 'max_min.csv')
    else:
        print('MDA skipped: restart was False')

    # load extreme rate, used to quantify the RPs of damages when risk modeling
    extreme_rate = np.loadtxt('fitted_stats/extreme_rate.txt').item()
    folder_lst = [cur_folder] # initialize the file paths where metrics will be saved for each output
    for i in range(len(output)):
        folder_lst.append(cur_folder + output[i] + '/')

    # for print statement
    if tgp:
        samp = 'TGP'
    else:
        samp = 'GP'

    print('Running ' + samp + f' on {len(output)} outputs')
    sam_util.run_tgp(folder_lst, model_folder, total_samples, denorm_subset,
                     minmax_scaler, extreme_rate, output = output, tgp = tgp,
                     num_sims = stoc_set.shape[0], MDA_ran = True, bonus = bonus, artificial_tide=artificial_tide)
    return None


def mda_tgp_samp(stoc_set, model_folder, cur_folder, total_samples = 100,
                 samples = 1, seed = None, mda_runs = None, tgp = True):
    '''
    Function to use when all artificial floods have been simulated.
    This is used in the stop_crit folder when the training datasets are investigated.
    The main difference with mda_tgp is that the coupled SFINCS and FIAT models
    do not need to be run after each sample is picked by the TGP as these are already
    known. The function is longer as some raw code from the run_tgp function (sampling_utils.py)
    is duplicated to not run SFINCS and FIAT.

    Parameters
    ---------
    stoc_set: str or Pandas.DataFrame
        Path to event set, or Pandas.DataFrame which contains the event set
        Contains the complete stochastic event set for an arbitrary number of dimensions
        in the real space.
    model_folder: str
        Path at which SFINCS, FIAT, damages and times will be saved
    cur_folder: str
        Path to which statistics created by MDA and TGP will be saved.
    total_samples: int
        Default: 100. If the number of simulations exceeds this value, the sampling will stop
    samples: int
        Default: 1. Number of artifical floods chosen by the TGP everytime it is used.
    seed: int
        Index to initialize the MDA algorithm. Default is maximum of 'S Mag' Parameter
    mda_runs: int
        Number of MDA samples. Defualt is the number of verticies (i.e. number of dimensions squared)
    tgp: bool
        True by default. Set to False if a Gaussian Process is wanted instead (faster, but does not partition)
        Will use TGP blackbox in R. 

    Output
    -------
    None. Important files in cur_folder are:
        stats.csv: contains highest uncertainty, time for each (T)GP model
        sampled_events: contains the sampled events and their normalized output.
    '''
    if type(stoc_set) == str:
        stoc_set = pd.read_csv(stoc_set)

    # expects stoc set to be in the real space
    
    minmax_scaler = normalizer.scaler(stoc_set)
    # stoc set has each dimension scaled between 0 and 1, less bias for MDA and TGP
    df_minmax_scaled = normalizer.normalize_dataset(stoc_set, minmax_scaler)

    # initialization of MDA, default is maximum of surge magnitude (first dimension)
    if seed is None:
        seed = df_minmax_scaled['S Mag'].argmax()

    # size of the subset of the MDA
    if mda_runs is None:
        mda_runs = 2**stoc_set.shape[1]

    # initialize a memory of the indices of the artificial floods
    # used to retrieve the damages associated with an artifical flood
    ind_array = np.arange(0, len(stoc_set), 1)

    # Based on initialized, event set, and size of subset, create MDA subset
    maxmin_class = selector_mda.MaxMin()
    lst_ind = maxmin_class.select_from_cluster(df_minmax_scaled.values, mda_runs, seed)
    subset = df_minmax_scaled.iloc[lst_ind].copy(deep = True)

    # remove indices of MDA subset from memory
    ind_array = np.delete(ind_array, subset.index)
    sampled_ind = subset.index.to_list()

    # read damages of artificial floods which have been previously simulated    
    target = pd.read_csv('Models/' + model_folder + '/damages.csv', index_col = 0)
    target_scaler = normalizer.scaler(target)
    # normalize the targets using a minmax scaler. Easier for the TGP.
    target_scaled = normalizer.normalize_dataset(target, target_scaler)

    # df_sampled: artifical floods the are used by the TGP to fit the surrogate model
    df_sampled = subset.copy()
    out_var = target.columns[0]
    # retrieve damages associated with df_sampled (currently only contains MDA subset)
    df_sampled[target.columns[0]] = target_scaled.iloc[subset.index].values
    # create df_available: artifical floods that are part of the event set but not used by the TGP
    df_available = df_minmax_scaled.drop(subset.index).reset_index(drop=True)
    # target_scaled: minmax scaled total damages associated with artificial floods in df_available
    target_scaled = target_scaled.drop(subset.index).reset_index(drop = True)

    # save df_sampled and df_available for the TGP to use
    use.create_empty_folder(cur_folder)
    df_sampled.to_csv(cur_folder + 'sampled_events.csv')
    df_available.to_csv(cur_folder + 'sample_space.csv')

    # choose which R executable to use based on TGP. If true will run TGP, else will run GP
    if tgp:
        r_exec = 'Notebooks/Scripts/R_scripts/tgp_exec.R'
    else:
        r_exec = 'Notebooks/Scripts/R_scripts/gp_exec.R'

    time_list = []
    alm_stats = []
    clock = mda_runs
    # create inputs for running the TGP in R
    for i in range(total_samples - clock):
        sol_file = cur_folder + 'sampled_events.csv'
        opt_file = cur_folder + 'sample_space.csv'

        temp_folder = cur_folder + str(clock) + '/'
        use.create_empty_folder(temp_folder)

        acqui_file = temp_folder + 'acqui.csv'
        X_file = temp_folder + 'X_mean.csv'
        XX_file = temp_folder + 'XX_mean.csv'
        plot_file = temp_folder + 'Plot.pdf'

        X5_file = temp_folder + 'X_5.csv'
        X95_file = temp_folder + 'X_95.csv'
        XX5_file = temp_folder + 'XX_5.csv'
        XX95_file = temp_folder + 'XX_95.csv'

        # string with all files that will be opened in R
        run_r_exec = ('Rscript ' + r_exec + ' ' + sol_file + ' ' + opt_file + ' ' + acqui_file + ' ' + 
                    X_file + ' ' + XX_file + ' ' + plot_file + ' ' + out_var + ' ' + X5_file + ' ' + 
                    X95_file + ' ' + XX5_file + ' ' + XX95_file + ' ' + 'bflat')

        df_sampled = pd.read_csv(cur_folder + 'sampled_events.csv', index_col = 0)
        df_available = pd.read_csv(cur_folder + 'sample_space.csv', index_col = 0)

        start_time = time.time()
        subprocess.run(run_r_exec, shell=True, capture_output=True, text=True) # run TGP in R
        end_time = time.time()
        execute_time = end_time - start_time # time the TGP
        time_list.append(execute_time)

        if subset.shape[1] == 2: # i.e. can still plot
            folder_figs = cur_folder + 'ALM/'
            # save figures into dedicated folder
            use.create_empty_folder(folder_figs)
            images = convert_from_path(plot_file)
            save_path_img = folder_figs + 'plot' + str(len(df_sampled)) + '.png'
            images[0].save(save_path_img)

        # read the alm statistics for available artifical floods, store the indicies and ALM
        # associated with the largest value
        df_alm = pd.read_csv(acqui_file)
        df_alm = df_alm.sort_values(by = 'x')
        indicies = df_alm.index[-samples:]
        alm_imp = df_alm.iloc[-samples:].values

        # obtain original index of artifical flood which has largest ALM
        true_ind = ind_array[indicies]
        # remove index from memory
        ind_array = np.delete(ind_array, indicies)
        
        # store alm and original index
        for i in range(samples):
            alm_stats.append(alm_imp[0][i])
            sampled_ind.append(true_ind[i])
        alm_list = alm_stats

        # save alm and time after each TGP to track how it evolves
        df_stats = pd.DataFrame({'High. Unc': alm_list,
                                 'Time [s]': time_list})
        df_stats.to_csv(cur_folder + 'stats.csv')

        # save original index of artifical floods that are used by TGP
        df_ind = pd.DataFrame({'Ind': sampled_ind})
        df_ind.to_csv(cur_folder + 'sampled_ind.csv')

        # add the new artifical floods chosen by the TGP to current subset used by the TGP
        gp_samples = df_available.iloc[indicies] # retrieve input(s)
        gp_consequences = target_scaled.iloc[indicies] # retrieve output(s)
        for j in range(samples):
            new_row = np.concatenate((gp_samples.values[j], gp_consequences.values[j]), axis = 0).reshape(1, -1)
            new_row_df = pd.DataFrame(new_row, columns = df_sampled.columns)
            df_sampled = pd.concat([df_sampled, new_row_df], ignore_index=True)
            clock += 1

        # remove new indicies by TGP from df_available and target_scaled
        df_available = df_available.drop(indicies).reset_index(drop = True)
        target_scaled = target_scaled.drop(indicies).reset_index(drop = True)

        # overwrite df_sample and df_available for new iteration of TGP
        df_sampled.to_csv(cur_folder + 'sampled_events.csv')
        df_available.to_csv(cur_folder + 'sample_space.csv')
    return None

def tgp_rerun(sampled, test, start_samples, cur_folder, out_var = 'Total'):
    '''
    Function used to rerun the TGP on a different event set. Will rerun the TGP
    len("sampled") - "start_samples" times.
    Used when the a posteriori method is applied on the test event set (10,000)
    and its accuracy is computed on the training event set (500). Only used in:
    Notebooks/Sampling/Compare/rerun_tgp.ipynb

    Parameters
    ---------
    sampled: pandas.DataFrame
        Final subset of artifical floods when TGP was used on a first event set
    test: pandas.DataFrame
        Second event set on which the TGP is wished to be used
    start_samples: int
        The number of samples of "sampled" the TGP will first use to obtain 
        the surrogate damage model for "test".
    cur_folder: str
        Path to which statistics created by MDA and TGP will be saved.
    out_var: str
        Default: 'Total'. Name of the output the TGP will be applied to.

    Output
    -------
    None. Important files in cur_folder are:
        stats.csv: contains highest uncertainty for each (T)GP model
        Important files in sub folders are:
            XX_mean.csv: contains the normalized damages associated with
            the "test" input at each TGP iteration. When compared with the 
            damages obtain from SFINCS and FIAT, an accuracy metric can be
            used.
    '''
    total_samples = len(sampled) + 1
    samples = 1 # number of samples that are chosen by the TGP per iteration
    alm_stats = []
    # test is saved to csv so that the TGP can estimate damages for its artifical floods
    use.create_empty_folder(cur_folder)
    test.to_csv(cur_folder + 'sample_space.csv')
    # run the TGP multiple times to estimate the gain in accuracy as the number of artifical floods
    # increases. Save the necessary files for this process
    for i in range(total_samples - start_samples):
        print(i + 4)
        sampled.iloc[:(start_samples + i)].to_csv(cur_folder + 'sampled_events.csv')
        sol_file = cur_folder + 'sampled_events.csv'
        opt_file = cur_folder + 'sample_space.csv'

        temp_folder = cur_folder + str(i) + '/'
        use.create_empty_folder(temp_folder)

        acqui_file = temp_folder + 'acqui.csv'
        X_file = temp_folder + 'X_mean.csv'
        XX_file = temp_folder + 'XX_mean.csv'
        plot_file = temp_folder + 'Plot.pdf'

        X5_file = temp_folder + 'X_5.csv'
        X95_file = temp_folder + 'X_95.csv'
        XX5_file = temp_folder + 'XX_5.csv'
        XX95_file = temp_folder + 'XX_95.csv'
        r_exec = 'Notebooks/Scripts/R_scripts/tgp_exec.R'

        run_r_exec = ('Rscript ' + r_exec + ' ' + sol_file + ' ' + opt_file + ' ' + acqui_file + ' ' + 
                    X_file + ' ' + XX_file + ' ' + plot_file + ' ' + out_var + ' ' + X5_file + ' ' + 
                    X95_file + ' ' + XX5_file + ' ' + XX95_file + ' ' + 'bflat')

        subprocess.run(run_r_exec, shell=True, capture_output=True, text=True) # run the TGP

        # save graphical representation of the TGP
        if sampled.shape[1] == 2: # i.e. can still plot
            folder_figs = cur_folder + 'ALM/'
            use.create_empty_folder(folder_figs)
            images = convert_from_path(plot_file)
            save_path_img = folder_figs + 'plot' + str(i) + '.png'
            images[0].save(save_path_img)

        # ALM only used as a monitoring metric. It is not used to pick the next artificial flood
        # This is because the next artifical flood is contained in the "sampled" input
        df_alm = pd.read_csv(acqui_file)
        df_alm = df_alm.sort_values(by = 'x')
        alm_imp = df_alm.iloc[-samples:].values
        
        for i in range(samples):
            alm_stats.append(alm_imp[0][i])
        alm_list = alm_stats

        df_stats = pd.DataFrame({'High. Unc': alm_list})
        df_stats.to_csv(cur_folder + 'stats.csv')
    return None
