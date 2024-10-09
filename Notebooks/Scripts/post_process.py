import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.patches as mpatches

import Notebooks.Scripts.normalization as normalizer
import Notebooks.Scripts.sampling_utils as sam_util

# Postprocess risk curves, dominance bar plots, ks and MWU tests between multiple dimensions

def collect_folders(model_type, dims, output):
    folders = []
    for i in dims:
        folders.append('Notebooks/4.Active_learning/' + str(i) + 'd_' + model_type + '/' + output + '/')
    return folders

def collect_sub_folders(model_type, multi_out, dim = 2):
    folders = []
    for classif in multi_out:
        folders.append('Notebooks/4.Active_learning/' + str(dim) + 'd_' + model_type + '/' + classif + '/')
    return folders

def plot_all_ead(model_type = 'single', dims = [2, 3, 4, 5, 6], output = 'Total'):
    folders = collect_folders(model_type, dims, output)
    clock = dims[0]
    for storage in folders:
        df_stats = pd.read_csv(storage + 'stability.csv', index_col = 0)
        sam_util.plot_ead(df_stats, clock)
        clock += 1

def plot_risk_curves(model_type = 'single', dims = [2, 3, 4, 5, 6], output = 'Total'):
    ex_rate = np.loadtxt('fitted_stats/extreme_rate.txt').item()
    folders = collect_folders(model_type, dims, output)

    stats_list = []
    dynamic_handles = []
    clock = dims[0]
    for storage in folders:
        df_sampled = pd.read_csv(storage + 'sampled_events.csv', index_col = 0)
        dimensions = df_sampled.columns[:clock].to_list() # sampled contains output as well
        
        damages = pd.read_csv('Models/TGP_' + str(clock) + 'd_' + model_type + '/damages.csv', index_col = 0).iloc[:len(df_sampled)]
        max_cons = damages[output].max()
        min_cons = damages[output].min()

        ead, mean_df = sam_util.obtain_cons_curve(storage + 'X_mean.csv', storage + 'XX_mean.csv', min_cons, max_cons, ex_rate)
        stats_list.append(mean_df)
        _, five_df = sam_util.obtain_cons_curve(storage + 'X_5.csv', storage + 'XX_5.csv', min_cons, max_cons, ex_rate)
        _, ninefive_df = sam_util.obtain_cons_curve(storage + 'X_95.csv', storage + 'XX_95.csv', min_cons, max_cons, ex_rate)
        if clock == dims[0]:
            ax_2 = sam_util.plot_fits(mean_df, ead, dimensions, cons_five = five_df, cons_95 = ninefive_df)
        else:
            _ = sam_util.plot_fits(mean_df, ead, dimensions, ax_cons = ax_2, cons_five = five_df, cons_95 = ninefive_df)
        
        # creating custom patch for each risk curve
        rect_i = mpatches.Patch(color=sam_util.color_dic(clock), alpha=0.4, label=f'd = {clock} (EAD = {ead/1e6:.2f} mil. USD)')
        dynamic_handles.append(rect_i)
        
        clock += 1
    
    # static legend entries
    grey_dot = plt.Line2D([0], [0], marker='o', color='w', label='Synthetic events for d dimensions',
                      markerfacecolor='grey', markersize=15)
    large_rect = mpatches.Patch(color='grey', label='TGP-LLM uncertainty for d dimensions')

    # Creating the custom legend with both static and dynamic entries
    custom_handles = [grey_dot, large_rect] + dynamic_handles
    plt.legend(handles=custom_handles, loc='upper left')

    _, top = plt.ylim()
    plt.ylim([0, top])

    if output == 'Total':
        plt.savefig('Figures/PDF/f05.pdf', dpi=300, format='pdf', bbox_inches="tight")
        plt.savefig('Figures/PNG/f05.png', format='png', bbox_inches="tight")
    return stats_list

def plot_dominance(model_type = 'single', dims = [2, 3, 4, 5, 6], output = 'Total', multi_out = None):
    ex_rate = np.loadtxt('fitted_stats/extreme_rate.txt').item()
    p_exrate1 = np.loadtxt('fitted_stats/p_mag_ex1.txt').item()
    ss_exrate1 = np.loadtxt('fitted_stats/ss_mag_ex1.txt').item()

    if multi_out is None:
        folders = collect_folders(model_type, dims, output)
        index = dims
    else:
        index = multi_out
        folders = collect_sub_folders(model_type, multi_out)
    dominance = pd.DataFrame(np.zeros((len(index), 2)), columns = ['Surge', 'Precipitation'], index = index)

    track = 0
    for seperate in index:
        sub_storage = folders[track]
        df_sampled = pd.read_csv(sub_storage + 'sampled_events.csv', index_col = 0)

        if multi_out is None:
            dimension = seperate
        else:
            dimension = dims[0]
            output = seperate
            print(output)
        
        damages = pd.read_csv('Models/TGP_' + str(dimension) + 'd_' + model_type + '/damages.csv', index_col = 0).iloc[:len(df_sampled)]
        max_cons = damages[output].max()
        min_cons = damages[output].min()

        _, mean_df = sam_util.obtain_cons_curve(sub_storage + 'X_mean.csv', sub_storage + 'XX_mean.csv', min_cons, max_cons, ex_rate)
        mean_df['Return Period'] = 1 / mean_df.iloc[:, 1]

        dim = pd.read_csv('fitted_stats/' + str(dimension) + 'd_sims.csv')
        if dimension == 6:
            dim = pd.read_csv('fitted_stats/' + str(dimension) + 'd_sims.csv', index_col = 0)
        scaler = normalizer.scaler(dim)

        all_events = pd.concat((
                                pd.read_csv(sub_storage + 'sampled_events.csv', index_col=0).iloc[:, :dimension],
                                pd.read_csv(sub_storage + 'sample_space.csv', index_col=0)
                                )).reset_index(drop = True)
        
        all_events = normalizer.denormalize_dataset(all_events, scaler)
        all_events.columns = dim.columns
        extreme_events = all_events.iloc[mean_df[mean_df['Return Period'] > 100].index]
        all_events['Return Period (RP)'] = 'RP < 100 Years'
        all_events['Return Period (RP)'].iloc[extreme_events.index] = 'RP > 100 Years'
        if dimension < 3:
            all_events = all_events
        else:
            all_events = all_events.iloc[:, [0, 1, 2, -1]]
        dominance.loc[seperate, 'Surge'] = sum(extreme_events['S Mag [m]'] > ss_exrate1)/len(extreme_events) # thresholds based on POT
        dominance.loc[seperate, 'Precipitation'] = sum(extreme_events['P Mag [mm/hr]'] > p_exrate1)/len(extreme_events) # thresholds based on POT
        sns.pairplot(all_events, hue = 'Return Period (RP)', height = 1.5, aspect = 2,
                    palette= {'RP > 100 Years': "red", 'RP < 100 Years': "black"},
                    corner=True, diag_kind=None, plot_kws={'s': 3})
        plt.suptitle(seperate)
        track += 1
    return dominance

def ecdf(values):
    sorted_data = np.sort(values)
    n = len(sorted_data)
    x = np.unique(sorted_data)
    edf = np.searchsorted(sorted_data, x, side='right') / n
    return x, edf

def plot_ks(stats_list, type = 'Total'):
    num_dims = len(stats_list)

    fig, axes = plt.subplots(num_dims, num_dims - 1, sharex=False, sharey=False, figsize=((num_dims - 1)*4,num_dims*4), gridspec_kw={'hspace':0.0, 'wspace':0.0})
    start_lst = []
    end_lst = []
    for i in range(4):
        df_simp = stats_list[i]
        for j in range(5):
            if j > i:
                df_comp = stats_list[j]
                axes[j, i].plot(ecdf(df_simp.iloc[:, 0].values)[0], ecdf(df_simp.iloc[:, 0].values)[1], c = sam_util.color_dic(i+2), label = f'{i+2} dimensions')
                axes[j ,i].plot(ecdf(df_comp.iloc[:, 0].values)[0], ecdf(df_comp.iloc[:, 0].values)[1], c = sam_util.color_dic(j+2), label = f'{j+2} dimensions')

                start = df_comp.iloc[:, 0].values.min()
                ind_stop_comp = np.where(ecdf(df_comp.iloc[:, 0].values)[1] > 0.95)[0][0]
                end = df_comp.iloc[:, 0].values[len(df_comp) - ind_stop_comp]

                start_lst.append(start)
                end_lst.append(end)

                axes[j, i].set_xlim([start, end])
                axes[j, i].set_xlabel(type + ' Economic Damages [USD]', fontsize = 12)
                ks_test = scipy.stats.ks_2samp(df_simp.iloc[:, 0].values, df_comp.iloc[:, 0].values, method = 'asymp')
                axes[j, i].axvline(ks_test.statistic_location, ls = ':', c = 'k', label = f'stat loc (result: {ks_test.statistic:.2f})')
                axes[j, i].legend(fontsize = 12)
                axes[j, i].text(start, 0.98, f"KS p-value = {ks_test.pvalue:.4f}", ha='left', va='bottom', fontsize = 12)
                if i == 0:
                    axes[j, i].set_ylabel('Cumulative probability [-]', fontsize = 12)
                else:
                    axes[j, i].set_yticklabels('')
                if j == 4:
                    axes[j, i].set_xlabel(type + ' damages [USD]', fontsize = 12)
                else:
                    axes[j, i].set_xticklabels('')
            else:
                axes[j, i].set_visible(False)
    for z in range(5):
        for i in range(4):
            axes[z, i].set_xlim([np.array(start_lst).min(), np.array(end_lst).max()])
    if type == 'Total':
        plt.savefig('Figures/PDF/f10.pdf', dpi=300, format='pdf', bbox_inches="tight")
        plt.savefig('Figures/PNG/f10.png', format='png', bbox_inches="tight")
    return None

def get_return_periods(x, a=0.0, extremes_rate=1.0):
    assert np.all(np.isfinite(x))
    b = 1.0 - 2.0 * a
    ranks = (len(x)+1) - scipy.stats.rankdata(x, method="average")
    freq = ((ranks - a) / (len(x) + b)) * extremes_rate
    rps = 1 / freq
    return rps

def get_boot_pop(gp_conseq, n_boot):
    dim = gp_conseq['Conseq'].values
    n_obs = len(dim)
    bootstrap = np.random.choice(dim, n_obs * n_boot)
    return bootstrap

def get_boot_ead(gp_conseq, ex_rate, n_boot = 500):
    bootstrap_ead = []
    
    n_obs = len(gp_conseq)
    bootstrap = get_boot_pop(gp_conseq, n_boot)
    for i in range(n_boot - 1):
        sub_sample = bootstrap[i * n_obs:(i + 1) * n_obs]
        obs_rps = get_return_periods(sub_sample, extremes_rate = ex_rate)
        df_risk = pd.DataFrame({'Conseq': sub_sample,
                                'Prob': 1/obs_rps})
        df_risk = df_risk.sort_values('Prob')
        ead = scipy.integrate.trapezoid(y = df_risk.iloc[:, 0].values, x = df_risk.iloc[:, 1].values)
        bootstrap_ead.append(ead)
    bootstrap_ead = np.array(bootstrap_ead)
    return bootstrap_ead

def plot_mwu(stats_list, type = 'Total', n_boot = 500):
    num_dims = len(stats_list)
    ex_rate = np.loadtxt('fitted_stats/extreme_rate.txt').item()

    fig, axes = plt.subplots(num_dims, num_dims - 1, sharex=False, sharey=False, figsize=((num_dims - 1)*4, num_dims*4), gridspec_kw={'hspace':0.0, 'wspace':0.0})

    boot_lst = []
    for y in range(5):
        boot_lst.append(get_boot_ead(stats_list[y], ex_rate = ex_rate, n_boot = n_boot))
    min = []
    max = []
    for i in range(4):
        simp = boot_lst[i]
        for j in range(5):
            if j > i:
                comp = boot_lst[j]

                min.append(simp.min())
                min.append(comp.min())
                max.append(simp.max())
                max.append(comp.max())

                u_test = scipy.stats.mannwhitneyu(simp, comp)

                axes[j, i].hist(simp, color = sam_util.color_dic(i+2), label = f'{i+2} dimensions', alpha = 0.5)
                axes[j, i].hist(comp, color = sam_util.color_dic(j+2), label = f'{j+2} dimensions', alpha = 0.5)
                axes[j, i].legend(fontsize = 12, loc = 'lower center')
                if i == 0:
                    axes[j, i].set_ylabel('Count [-]', fontsize = 12)
                else:
                    axes[j, i].set_yticklabels('')
                if j == 4:
                    for z in range(5):
                        axes[z, i].set_xlim([np.array(min).min(), np.array(max).max()])
                    axes[j, i].set_xlabel(type + ' EAD [USD]', fontsize = 12)
                else:
                    axes[j, i].set_xticklabels('')
                axes[j, i].text(0.01, 0.99, f'MWU p-value = {u_test.pvalue:.4f}', 
                                transform=axes[j, i].transAxes, fontsize=12, 
                                verticalalignment='top', horizontalalignment='left')
            else:
                axes[j, i].set_visible(False)
    if type == 'Total':
        plt.savefig('Figures/PDF/f11.pdf', dpi=300, format='pdf', bbox_inches="tight")
        plt.savefig('Figures/PNG/f11.png', format='png', bbox_inches="tight")

def collect_tgp_times(tgp_folder, classif = ['Total'], dimensions = [2, 3, 4, 5, 6]):
    '''
    For different dimensionalities and outputs, classify the time according to a certain component of a work flow
    '''
    empty_lst = []
    diff_comp = pd.DataFrame()
    samples_req = pd.DataFrame(np.zeros((len(classif), len(dimensions))), columns = dimensions, index = classif)
    if len(classif) == 1:
        post = 'single'
    elif len(classif) == 2:
        post = 'multi'
    else:
        post = 'sub'
    for i in dimensions:
        for cluster in classif:
            tgp_stats = pd.read_csv(tgp_folder + str(i) + 'd_' + post + '/' + cluster + '/stability.csv', index_col = 0)
            tgp_samples = pd.read_csv(tgp_folder + str(i) + 'd_' + post + '/' + cluster + '/sampled_events.csv', index_col = 0)
            num_samples = tgp_samples.shape[0]
            samples_req.loc[cluster, i] = np.round(num_samples, decimals=1)
            if cluster == classif[0]:
                 tgp_times = np.append(np.zeros(2**i - 1), tgp_stats['Time [s]'])
            else:
                tgp_times = np.append(tgp_times, tgp_stats['Time [s]'])
        times_tgp = pd.read_csv('Models/TGP_' + str(i) + 'd_' + post + '/times.csv', index_col = 0)

        clock = len(times_tgp[times_tgp[times_tgp.columns[0]] != 0])
        times_tgp = times_tgp.iloc[:clock]
        time_post = pd.DataFrame(tgp_times, columns = ['TGP'])
        pre_stop_times = pd.concat((times_tgp, time_post), axis = 1).reset_index(drop = True)
        empty_lst.append(pre_stop_times)
        diff_comp[f'A Posteriori ({i} dims)'] = empty_lst[i-2].sum()/60
    return diff_comp, samples_req

def plot_times(df_comp):
    num_bars = df_comp.shape[1]
    x_values = np.arange(1, num_bars + 1, 1)
    plt.figure(figsize = (8, 6))
    plt.bar(x_values, df_comp.loc['FIAT'], label='Delft-FIAT')
    plt.bar(x_values, df_comp.loc['SFINCS'], bottom=df_comp.loc['FIAT'], label='SFINCS')
    for i in range(num_bars - 1):
        i += 1
        if i ==1:
            label = 'TGP-LLM'
        else:
            label = ''
        plt.bar(x_values[i], df_comp.loc['TGP'].iloc[i], bottom=[df_comp.loc['FIAT'].iloc[i]+df_comp.loc['SFINCS'].iloc[i]], label=label, color= 'k')



    plt.xlabel('Approach used', fontsize = 12)
    plt.ylabel('Time [min]', fontsize = 12)
    # plt.title('Comparing Computational Cost of Sampling Algroithms for Complete', fontsize = 12)

    plt.xticks(x_values, df_comp.columns, fontsize = 12)

    plt.legend(fontsize = 12)
    plt.grid()
    return None


