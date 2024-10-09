import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import Notebooks.Scripts.normalization as normalizer
import Notebooks.Scripts.sampling_utils as sam_util
import Notebooks.Scripts.post_process as post

def RMSE(sampled, target):
    rmse = np.sqrt(np.mean((sampled - target)**2))
    return rmse

def collect_metrics(stoc_set, folder, num_mda, ground_val, stop_crit = 0.1):
    ind_array = np.arange(0, len(stoc_set), 1)
    all_ind = pd.read_csv(folder + 'sampled_ind.csv', index_col=0)
    total_samples = len(all_ind)
    tgp_samples = total_samples - num_mda
    ex_rate = np.loadtxt('fitted_stats/extreme_rate.txt').item()

    scaler = normalizer.scaler(ground_val)
    ead_ground, risk_ground = sam_util.obtain_cons(ground_val.values, ex_rate = ex_rate)
    
    rmse_arr = np.zeros((2, tgp_samples))
    alm = np.zeros((2, tgp_samples))
    ead = np.zeros((3, tgp_samples))

    memory = np.ones((2, 1))

    ks_list = []
    risk_list = []
    cont = True
    for i in np.arange(num_mda, total_samples):
        sampled_mean = pd.read_csv(folder + str(i) + '/X_mean.csv').rename(columns = {'x': 'Total'})
        interp_mean = pd.read_csv(folder + str(i) + '/XX_mean.csv').rename(columns = {'x': 'Total'})

        sampled_5 = pd.read_csv(folder + str(i) + '/X_5.csv').rename(columns = {'x': 'Total'})
        interp_5 = pd.read_csv(folder + str(i) + '/XX_5.csv').rename(columns = {'x': 'Total'})

        sampled_95 = pd.read_csv(folder + str(i) + '/X_95.csv').rename(columns = {'x': 'Total'})
        interp_95 = pd.read_csv(folder + str(i) + '/XX_95.csv').rename(columns = {'x': 'Total'})

        alm[0, i - num_mda] = pd.read_csv(folder + str(i) + '/acqui.csv').mean()
        alm[1, i - num_mda] = pd.read_csv(folder + str(i) + '/acqui.csv').max()

        output_memory = memory
        bool = output_memory != 1.0
        stop_ind = len(output_memory[bool])
        if cont:
            if alm[0, i - num_mda] < stop_crit:
                memory[stop_ind] = alm[0, i - num_mda]
            else:
                memory[:,] = 1.0 # ensure that the values are consecutive

            if all(memory < stop_crit):
                ind = i
                cont = False

        sampled_ind = all_ind.iloc[:i].values.flatten()
        available_ind = np.delete(ind_array, sampled_ind)

        sampled = [sampled_mean, sampled_5, sampled_95]
        interp = [interp_mean, interp_5, interp_95]

        for z in range(3):
            sampled_d = normalizer.denormalize_dataset(sampled[z], scaler)
            interp_d = normalizer.denormalize_dataset(interp[z], scaler)

            pred_denorm = pd.concat((sampled_d, interp_d)).reset_index(drop=True).values.flatten()

            ead[z, i - num_mda], risk = sam_util.obtain_cons(pred_denorm, ex_rate = ex_rate)

            if z == 0:
                ks_test = scipy.stats.ks_2samp(risk.iloc[:, 0].values, risk_ground.iloc[:, 0].values, method = 'asymp')
                risk_list.append(risk.iloc[:, 0].values)
                ks_list.append(ks_test.pvalue)
                
                sampled_mean.index = sampled_ind
                interp_mean.index = available_ind

                sampled_d.index = sampled_ind
                interp_d.index = available_ind
                
                RMSE_sampled = RMSE(sampled_d, ground_val.iloc[sampled_ind])
                RMSE_remaining = RMSE(interp_d, ground_val.iloc[available_ind])
                rmse_arr[0, i - num_mda] = RMSE_sampled
                rmse_arr[1, i - num_mda] = RMSE_remaining

    return rmse_arr, alm, ead, ead_ground, risk_ground, risk_list, ks_list, ind

def get_plots(stoc_set, folder, num_mda, ground_val, stop_crit = 0.1):
    ex_rate = np.loadtxt('fitted_stats/extreme_rate.txt').item()

    all_ind = pd.read_csv(folder + 'sampled_ind.csv', index_col=0)
    tot_samples = len(all_ind)
    rmse_arr, alm, ead, ead_ground, risk_ground, risk_list, ks_list, ind = collect_metrics(stoc_set, folder, num_mda, ground_val, stop_crit)
    roll = pd.DataFrame(ead[0]).rolling(10, min_periods = 10)
    coef_var = ((roll.std().values)/(roll.mean().values))
    
    dim = stoc_set.shape[1]
    dim = str(dim) + '_'

    x = np.arange(num_mda, tot_samples)

    plt.plot(x, rmse_arr[0], label = 'Sampled')
    plt.plot(x, rmse_arr[1], label = 'Available')
    plt.axvline(ind, c = 'k', label = 'Proposed Stop')
    plt.legend(fontsize = 14)
    plt.ylabel('RMSE [USD]', fontsize = 14)
    plt.xlabel('Total Samples [-]', fontsize = 14)
    plt.grid()
    plt.xlim([num_mda - 2, tot_samples + 2])
    plt.savefig('Figures/Train/' + dim + 'rmse.png')
    plt.show()

    plt.plot(x, rmse_arr[0], label = 'Sampled')
    plt.plot(x, rmse_arr[1], label = 'Available')
    plt.axvline(ind, c = 'k', label = 'Proposed Stop')
    plt.legend(fontsize = 14)
    plt.ylabel('RMSE [USD]', fontsize = 14)
    plt.xlabel('Total Samples [-]', fontsize = 14)
    plt.grid()
    plt.xlim([num_mda - 2, tot_samples + 2])
    plt.ylim([0, 1e7])
    plt.show()

    plt.plot(x, alm[0], label = 'ALM mean')
    plt.plot(x, alm[1], label = 'ALM max')
    plt.axvline(ind, c = 'k', label = 'Proposed Stop')
    plt.legend(fontsize = 14)
    plt.ylabel('ALM statistic [-]', fontsize = 14)
    plt.xlabel('Total Samples [-]', fontsize = 14)
    plt.grid()
    plt.xlim([num_mda - 2, tot_samples + 2])
    plt.savefig('Figures/Train/' + dim + 'ALM.png')
    plt.show()

    plt.plot(x, alm[0], label = 'ALM mean')
    plt.plot(x, alm[1], label = 'ALM max')
    plt.axvline(ind, c = 'k', label = 'Proposed Stop')
    plt.legend(fontsize = 14)
    plt.ylabel('ALM statistic [-]', fontsize = 14)
    plt.xlabel('Total Samples [-]', fontsize = 14)
    plt.grid()
    plt.xlim([num_mda - 2, tot_samples + 2])
    plt.ylim([0, 0.1])
    plt.show()

    plt.plot(x, ead[0], label = 'TGP mean')
    plt.fill_between(x, ead[1], ead[2], color = 'r', alpha=0.3, label = 'TGP Uncertainty')
    plt.axhline(ead_ground, c = 'g', ls = ':', label = 'Benchmark')
    plt.axvline(ind, c = 'k', label = 'Proposed Stop')
    plt.legend(fontsize = 14)
    plt.ylabel('EAD [USD]', fontsize = 14)
    plt.xlabel('Total Samples [-]', fontsize = 14)
    plt.grid()
    plt.xlim([num_mda - 2, tot_samples + 2])
    plt.ylim([ead_ground - 1e8, ead_ground + 1e8])
    plt.savefig('Figures/Train/' + dim + 'EAD.png')
    plt.show()

    plt.plot(x, coef_var, label = 'TGP mean')
    plt.axvline(ind, c = 'k', label = 'Proposed Stop')
    plt.legend(fontsize = 14)
    plt.ylabel('Coefficient of Variation [-]', fontsize = 14)
    plt.xlabel('Total Samples [-]', fontsize = 14)
    plt.grid()
    plt.xlim([num_mda - 2, tot_samples + 2])
    plt.savefig('Figures/Train/' + dim + 'COV.png')
    plt.show()

    plt.plot(x, ks_list, label = 'TGP mean')
    plt.axhline(0.05, ls = ':', label = 'Confidence Limit')
    plt.axvline(ind, c = 'k', label = 'Proposed Stop')
    plt.legend(fontsize = 14)
    plt.ylabel('KS p-value [-]', fontsize = 14)
    plt.xlabel('Total Samples [-]', fontsize = 14)
    plt.grid()
    plt.xlim([num_mda - 2, tot_samples + 2])
    plt.savefig('Figures/Train/' + dim + 'pvalue.png')
    plt.show()

    df_3 = risk_ground
    ind = ind - num_mda
    dim = str(stoc_set.shape[1]) + '_'

    plt.plot(post.ecdf(risk_list[ind])[0], post.ecdf(risk_list[ind])[1], c = 'b', label = 'TGP')
    plt.plot(post.ecdf(df_3.iloc[:, 0].values)[0], post.ecdf(df_3.iloc[:, 0].values)[1], c = 'g', label = 'Ground')
    start = np.array([risk_list[ind].min(), df_3.iloc[:, 0].values.min()]).min()
    ind_stop_2 = np.where(post.ecdf(risk_list[ind])[1] > 0.99)[0][0]
    ind_stop_3 = np.where(post.ecdf(df_3.iloc[:, 0].values)[1] > 0.99)[0][0]
    end = np.array([risk_list[ind][len(risk_list[ind]) - ind_stop_2], df_3.iloc[:, 0].values[len(df_3) - ind_stop_3]]).min()

    plt.xlim([start, end])
    plt.xlabel('Total Economic Damages [USD]', fontsize = 14)
    plt.ylabel('Probability [-]', fontsize = 14)
    ks_test = scipy.stats.ks_2samp(risk_list[ind], df_3.iloc[:, 0].values, method = 'asymp')
    plt.axvline(ks_test.statistic_location, ls = ':', c = 'k', label = f'statistic location (result: {ks_test.statistic:.2f})')
    plt.legend(fontsize = 14)
    plt.text(start, 0.98, f"KS p-value = {ks_test.pvalue:.2f}", ha='left', va='bottom', fontsize = 14)
    plt.title(f'KS Test Statistic after {ind + num_mda} total samples', fontsize = 14)
    plt.savefig('Figures/Train/' + dim + 'ECDF.png')
    plt.show()

    return rmse_arr[1]

def stop_plots(stoc_set, folder, num_mda, ground_val, stop_crit = 0.1):
    ex_rate = np.loadtxt('fitted_stats/extreme_rate.txt').item()
    fig, axs = plt.subplots(4, 2, figsize=(11, 12))
    labels = [['(a)', '(b)', '(c)', '(d)'], ['(e)', '(f)', '(g)', '(h)']]
    for i in range(2):

        all_ind = pd.read_csv(folder[i] + 'sampled_ind.csv', index_col=0)
        tot_samples = len(all_ind)
        rmse_arr, alm, ead, ead_ground, risk_ground, risk_list, ks_list, ind = collect_metrics(stoc_set[i], folder[i], num_mda[i], ground_val[i], stop_crit)
        dim = stoc_set[i].shape[1]
        dim = str(dim) + '_'
        x = np.arange(num_mda[i], tot_samples)



        # Loop through each subplot and add a label
        axs[0, i].plot(x, rmse_arr[0], label = 'Simulated', zorder = 2)
        axs[0, i].plot(x, rmse_arr[1], label = 'Non-simulated', zorder = 1)
        axs[0, i].axvline(ind, c = 'k', label = 'Proposed stop', linewidth=1.5, zorder = 0)
        axs[0, i].legend(fontsize = 12)
        axs[0, i].set_ylabel('RMSE [USD]', fontsize = 12)
        axs[0, i].grid()
        axs[0, i].set_xlim([num_mda[i] - 2, tot_samples + 2])

        axs[1, i].plot(x, alm[0], label = 'ALM mean', zorder = 2)
        axs[1, i].plot(x, alm[1], label = 'ALM max', zorder = 1)
        axs[1, i].axvline(ind, c = 'k', label = 'Proposed stop', linewidth=1.5, zorder = 0)
        axs[1, i].legend(fontsize = 12)
        axs[1, i].set_ylabel('ALM statistic [-]', fontsize = 12)
        axs[1, i].grid()
        axs[1, i].set_xlim([num_mda[i] - 2, tot_samples + 2])

        axs[2, i].plot(x, ead[0], label = 'TGP-LLM mean', zorder= 2)
        axs[2, i].fill_between(x, ead[1], ead[2], color = 'r', alpha=0.3, label = 'TGP-LLM uncertainty')
        axs[2, i].axhline(ead_ground, c = 'g', ls = ':', label = 'Benchmark', zorder = 1)
        axs[2, i].axvline(ind, c = 'k', label = 'Proposed stop', linewidth=1.5, zorder = 0)
        axs[2, i].legend(fontsize = 12)
        axs[2, i].set_ylabel('EAD [USD]', fontsize = 12)
        axs[2, i].grid()
        axs[2, i].set_xlim([num_mda[i] - 2, tot_samples + 2])
        axs[2, i].set_ylim([ead_ground - 1e8, ead_ground + 1e8])

        axs[3, i].plot(x, ks_list, label = 'TGP-LLM mean', zorder = 2)
        axs[3, i].axhline(0.05, ls = '--', label = 'Confidence limit', zorder = 1, linewidth = 2)
        axs[3, i].axvline(ind, c = 'k', linewidth=1.5, label = 'Proposed stop', zorder = 0)
        axs[3, i].legend(fontsize = 12)
        axs[3, i].set_ylabel('KS p-value [-]', fontsize = 12)
        axs[3, i].set_xlabel('Total simulations [-]', fontsize = 12)
        axs[3, i].grid()
        axs[3, i].set_xlim([num_mda[i] - 2, tot_samples + 2])


        for j, ax in enumerate(axs[:, i]):
            ax.text(0.025, 0.95, labels[i][j], transform=ax.transAxes, fontsize=14, verticalalignment='top', zorder = 3, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.savefig('Figures/PDF/f07.pdf', dpi=300, format='pdf', bbox_inches="tight")
    plt.savefig('Figures/PNG/f07.png', format='png', bbox_inches="tight")

    # Show the plot
    plt.show()
