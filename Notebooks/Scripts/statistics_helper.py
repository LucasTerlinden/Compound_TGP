import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import hydromt
import json
import scipy.stats as sc
import seaborn as sns
import pyvinecopulib as pv
import matplotlib.gridspec as gridspec

from scipy.stats import kendalltau
from bisect import bisect_right
from Notebooks.Scripts.Useful import find_units
from Notebooks.Scripts.eva import plot_return_values
import matplotlib
from pyextremes import plot_mean_residual_life, plot_parameter_stability, plot_threshold_stability

## Script used to conduct statistical analysis, and plot results

return_period = np.array([1.01, 2, 5, 10, 20, 30, 50, 100])
window_size = 24*3 # 3 days
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=15)

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

def find_quantile(array, value):
    '''
    Obtain the quantile of a value
    '''
    sorted_array = sorted(array)
    count = bisect_right(sorted_array, value)
    quantile = (count / len(sorted_array)) * 100
    return quantile

def POT_brute_force(df_single, threshold_guess, decluster = 7*24):
    '''
    Given an initial guess, function will return
    extremes and threshold which has extreme rate close to 1
    Start large with guess, function reduces threshold per iteration.
    '''
    units, _ = find_units(df_single)
    year_span = (df_single.index.max() - df_single.index.min()).days / 365.25
    threshold_start = find_quantile(df_single.values, threshold_guess)
    threshold = threshold_start/100
    for i in range(1000):
        threshold = np.round(threshold, 5)
        print(f'Current Threshold (quantile): {threshold}')
        extremes = hydromt.stats.extremes.eva(da = create_da(df_single),
                                            ev_type='POT',
                                            min_dist = decluster,
                                            qthresh = threshold,
                                            rps = return_period,
                                            criterium = 'AIC')
        np_ex = extremes['peaks'].values
        peaks_year = len(np_ex[~np.isnan(np_ex)])/year_span
        if peaks_year < 1:
            print(f'Peaks per year on average: {peaks_year:.2f}')
            threshold -= 0.00005
        else:
            threshold_val = np.quantile(df_single.values, threshold)
            print('Breaking...\n'
                f'Peaks per year on average: {peaks_year:.2f}\n'
                f'Threshold value: {threshold_val:.2f} ' + units)
            break
    return extremes, threshold, threshold_val

def POT_threshold_plots(df_single):
    '''
    Use pyextremes
    '''
    plot_mean_residual_life(df_single)
    plot_parameter_stability(df_single)
    return None

def threshold_stable(df_single, rp, threshold_list, points = 20):
    '''
    Use pyextremes
    '''
    plot_threshold_stability(
        df_single,
        return_period = rp,
        thresholds = np.linspace(threshold_list[0], threshold_list[1], points),
    )
    return None

def plot_extremes(df_single, extremes, threshold = None, ev_type = 'BM'):
    extremes['peaks'].plot.scatter(c = "r", label = 'Extremes')
    units, ind = find_units(df_single)
    plt.plot(df_single.index,
         df_single.values,
         linewidth = 0.1, label = 'Observations')
    if ev_type == 'POT':
        threshold_value = np.quantile((df_single.values), threshold)
        plt.axhline(threshold_value, c = 'k', ls = ':',
            label = f'Threshold: {threshold_value:.2f}')
    elif ev_type == 'BM':
        None
    plt.legend(loc = 'lower right', fontsize = 14)
    plt.xlabel('Time')
    plt.ylabel(df_single.name[:ind[0]-1] + ' [' + units + ']')
    plt.title(ev_type + ' Analysis of ' + df_single.name[:ind[0]-1])
    plt.text(df_single.index[0], df_single.min(), f'Extreme Rate = {extremes["extremes_rate"].data:.2f}', fontsize = 14)   
    return None

def plot_fits(df_single, ds_extremes, return_p = return_period):
    df_extremes = pd.DataFrame({'peaks': ds_extremes['peaks'].values}).dropna()
    extremes = df_extremes['peaks'].values
    fit = ds_extremes['parameters']
    curve = ds_extremes['distribution'].item()
    rate = ds_extremes['extremes_rate'].values
    plot_return_values(extremes, fit, curve, alpha = 0.95, rps =  return_p, extremes_rate = rate)

    units, ind = find_units(df_single)
    plt.title(df_single.name[:ind[0] - 1] + 'EVA fit with ' + curve + ' curve')
    plt.ylabel('Return Value [' + units + ']')
    return None

def find_largest_within_window(peak_name, df_peaks, df_data, threshold, window_size=window_size, one_side = False):
    result = []
    name = df_data.name
    if df_peaks.index.tz is None:
        time = df_peaks.index.tz_localize('UTC')
    else:
        time = df_peaks.index.tz_convert('UTC') # not robust, if index has not been initialized in past

    count = 0
    for peak_time in time:
        if one_side:
            # only look ahead, useful for skew_surge and tide,
            # when it is known that skew_surge occurs with the following tide
            data_window = df_data[(df_data.index >= peak_time - pd.Timedelta(hours=1)) &
                                  (df_data.index <= peak_time + pd.Timedelta(hours=window_size))]
        else:
            # Filter data within the window around the peak time
            data_window = df_data[(df_data.index >= peak_time - pd.Timedelta(hours=window_size)) &
                                  (df_data.index <= peak_time + pd.Timedelta(hours=window_size))]

        # Find the row with the largest value
        
        max_row = data_window.argmax()
        if data_window[max_row] > threshold:
            compound = 'r'
        else:
            compound = 'k'
        result.append({
            'time_' + peak_name: peak_time,
            'value_' + peak_name: df_peaks.iloc[count],
            'value_' + name: data_window[max_row],
            'time_' + name: data_window.index[max_row],
            'Compound?': compound
        })
        count += 1

    return pd.DataFrame(result)

def find_durations(df_single, df_marginal, quantile, hours = window_size):
    '''
    no localize
    hours: int
        Shearch for duration hours before and after magnitude
    '''
    tidal_peaks = pd.read_csv('Data/skew_surge_tides.csv', parse_dates = ['DateTime(UTC)'])
    tidal_peaks.set_index('DateTime(UTC)', inplace = True)

    threshold = np.quantile(df_single, quantile)
    durations = []
    sk_left = []
    sk_right = []
    for i in range(len(df_marginal)):
        center = df_marginal.index[i]
        start = df_marginal.index[i] - np.timedelta64(hours, 'h')
        end = df_marginal.index[i] + np.timedelta64(hours, 'h')

        # initialize begin and stop if threshold is never reached
        begin = start
        stop = end
        df_ = df_single.loc[start:end]

        for z in range(hours):
            back = center - np.timedelta64(z, 'h')
            if df_.loc[back] < threshold:
                begin = back
                break
            else:
                continue

        for w in range(hours):
            forw = center + np.timedelta64(w, 'h')
            if df_.loc[forw] < threshold:
                stop = forw
                break
            else:
                continue
        if df_single.name == 'Skew_surge (m)':
            correction = np.timedelta64(6, 'h') # center about high tide
            ## skew surge length is dependant on length of tide:
            df_skew_un = df_single.loc[begin:stop].unique()
            num_cycles = len(df_skew_un[df_skew_un > 0.2])
            durations.append(num_cycles)
        else:
            correction = np.timedelta64(0, 'h')
            duration = np.abs(int((stop - begin).total_seconds()/3600))
            durations.append(duration)
        center = center + correction
        skew_left = np.abs(int((begin - center).total_seconds()/3600))
        sk_left.append(skew_left)
        skew_right = np.abs(int((stop - center).total_seconds()/3600))
        sk_right.append(skew_right)
    durations = np.array(durations)
    sk_left = np.array(sk_left)
    sk_right = np.array(sk_right)
    return durations, sk_left, sk_right

# pyvinecopulib only works with unity
def unity(data):
    M = data.shape[0]  # Reading number of observations per node
    ranks = data.rank(axis=0)
    u_hat = ranks / (M + 1)
    return u_hat

def twod_dep_plot(df_comp, column_ind, color = None, title = None):
    '''
    column_ind is a list of two indicies used to index the pandas
    '''
    ind1 = column_ind[0]
    ind2 = column_ind[1]

    tau, _ = kendalltau(df_comp.iloc[:, ind1], df_comp.iloc[:, ind2])
    col_names = df_comp.columns

    if color is not None:
        df_comp.plot.scatter(x = col_names[ind1], y = col_names[ind2], c = color)
        plt.text(plt.xlim()[0], plt.ylim()[1], r"$\tau$ = " + f"{tau:.2f}", fontsize=12, verticalalignment='top')
        plt.legend(handles=[plt.Line2D([], [], marker='o', linestyle='', color='r', label='Both Considered Extreme'),
                            plt.Line2D([], [], marker='o', linestyle='', color='k', label='Not Extreme')],
                            loc = 'upper right')
    else:
        df_comp.plot.scatter(x = col_names[ind1], y = col_names[ind2])

    if title is None:
        if col_names[ind1][:5] == 'value':
            title = 'when conditionalizing on: ' + col_names[ind1][6:]
        else:
            title = 'when conditionalizing on: ' + col_names[ind1]
    plt.title('Dependance Structure ' + title)
    plt.show()

def plot_pair(df, color = True, save_fig = False):
    '''
    If color is True, assumes this is the last column of the df
    '''
    plt.rc('legend',fontsize=15, title_fontsize=20)
    if color:
        all_tau = np.zeros((df.shape[1] - 1, df.shape[1] - 1))
        all_p = np.zeros((df.shape[1] - 1, df.shape[1] - 1))
        for i in range(df.shape[1] - 1):
            for j in range(df.shape[1] - 1):
                if i == j:
                    continue
                else:
                    tau, p = kendalltau(df.iloc[:, i], df.iloc[:, j])
                    all_tau[i, j] = tau
                    all_p[i, j] = p

        un = unity(df.iloc[:, :-1]).values
        vine_arr = np.concatenate((un, df.iloc[:, -1].values.reshape(-1, 1)), axis=1)

        unitless = [idx[:5] for idx in df.columns[:-1]] + [df.columns[-1]]
        unitless_col = df.columns.to_series().index.__class__(unitless)    
        unity_df = pd.DataFrame(vine_arr, columns = unitless_col)

        pairplot = sns.pairplot(unity_df, hue = 'Legend', height = 1.5, aspect = 2,
                                palette= {"Extreme Precipitation": "red", "Non Extreme Precipitation": "black"},
                                corner=True, diag_kind='hist', plot_kws={'s': 15})

        pairplot.fig.suptitle('Scatter matrix of the uniform empirical margins', y = 1.01, fontsize = 20)
        count = 0
        count_y = 0
        for ax in pairplot.axes.flat:
            if count < count_y:
                ax.text(0, 1, r'$\tau$=' + f" {all_tau[count, count_y]:.2f}", fontsize = 15,
                        transform=ax.transAxes, va='top', ha='left')
            else:
                None
            if count == (unity_df.shape[1] - 2): # takes into account 0 based indexing and the color column
                count = 0
                count_y += 1
            else:
                count +=1
    if save_fig:
        plt.savefig('Figures/PairPlot.png')
    return unity_df, all_p

def create_nested_dict(distribution_name, param_names, param_values):
    dist_conv = {'exp': 'expon',
                 'gpd': 'genpareto'}
    params = {param_name: param_values[i] for i, param_name in enumerate(param_names)}
    return {dist_conv[distribution_name]: params}

def save_marginal(file, marginal):
    with open(file, 'w') as json_file:
        json.dump(marginal, json_file, indent=4)
    return None

def load_marginal(file):
    with open(file, 'r') as json_file:
        marginal = json.load(json_file)
    return marginal

def obtain_freindly_marginal(marginal):
    '''
    marginal is a nested dictionary
    Return list of parameters, making it easier to index
    Return scipy stats distribution class, making it easier to call
    '''
    marginal_params = []
    for k in marginal.keys():
        param_dic = marginal[k]
        dist_name = k
        for sub_k in param_dic.keys():
            if param_dic[sub_k]!= 0:
                marginal_params.append(param_dic[sub_k])
    marginal_dist = getattr(sc, dist_name)
    if dist_name.startswith('trunc'):
        if len(marginal_params) == 5:
            a = param_dic['a'] * marginal_params[-1] + marginal_params[-2]
            b = param_dic['b'] * marginal_params[-1] + marginal_params[-2]

            a = -72
            b = 0
        elif dist_name == 'truncpareto':
            a = marginal_params[-1] + marginal_params[-2]
            b = param_dic['c'] * marginal_params[-1] + marginal_params[-2]
        else:
            a = param_dic['a']
            b = param_dic['b']
        lower_bound_cdf = marginal_dist.cdf(a, *marginal_params[:-2],
                                            loc=marginal_params[-2], scale=marginal_params[-1])
        upper_bound_cdf = marginal_dist.cdf(b, *marginal_params[:-2],
                                            loc=marginal_params[-2], scale=marginal_params[-1])
        extra = [lower_bound_cdf, upper_bound_cdf]
    else:
        extra = []
    return marginal_params, marginal_dist, extra

def fit_copulas(df_copula):
    '''
    Based on AIC test

    df_copulas has to have four columns, first two are the drivers in un ranked form
    last two are the drivers in ranked form
    '''
    cop_fam = [pv.BicopFamily(x) for x in np.arange(0,11)]

    bivariate_rank = np.empty( [len(df_copula),2] )

    bivariate_rank[:,0] = df_copula[df_copula.columns[2]]
    bivariate_rank[:,1] = df_copula[df_copula.columns[3]]

    bic_cop = []

    for f in np.arange( len(cop_fam) ):

        cop_temp = pv.Bicop(cop_fam[f])
        cop_temp.fit(data = bivariate_rank)
        bic_cop.append(cop_temp.bic(u = bivariate_rank)) #

    bic_cop = np.asarray(bic_cop)

    for i, cop in enumerate(cop_fam):
        copulas_name = cop_fam[i]  # Get the name of the distribution
        bic = bic_cop[i]
        print(f"Copula: {copulas_name}, BIC: {bic:.4f}")

    ## ---- FITTING OF THE BEST COPULA ----     
    ft_cop = np.where(bic_cop == np.min(bic_cop))[0][0]

    cop_biv_rank = pv.Bicop(cop_fam[ft_cop] )
    cop_biv_rank.fit(data = bivariate_rank)

    print('Best fitted Copula on data (based on BIC test)')
    print(cop_biv_rank)
        
    return cop_biv_rank

def obtain_sim_values(quantiles, marginal, threshold = [0, 1_000]):
    '''
    Function fits a theroetical distribution with nested dictionary that contains family distribution and respective parameters

    marginal is a nested dictionary, or an array (if emperical)
    quanitles is array like
    '''
    if isinstance(marginal, np.ndarray):
        sim_val = np.quantile(marginal, quantiles)
    else:
        params, dist_marg, extra = obtain_freindly_marginal(marginal)
        if len(extra) != 0:
            quantiles = extra[0] + quantiles * (extra[1] - extra[0])
        else:
            None
        sim_val = dist_marg.ppf(quantiles, *params[:-2], loc = params[-2], scale = params[-1])
        sim_val = np.maximum(threshold[0], sim_val)
        sim_val = np.minimum(threshold[1], sim_val)
    return sim_val

def plot_copula_2d(ranked_sim, df_sim, df_copula):
    '''

    '''
    plt.rc('legend',fontsize=15, title_fontsize=15)
    # PLOT Best copula - AIC
    fig = plt.figure(figsize = (10,5))
    gs = gridspec.GridSpec(ncols = 2, nrows = 1, figure = fig)

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])

    # ---------- Copula Domain
    ax11.scatter(ranked_sim[:,0], ranked_sim[:,1], s=20, alpha = 0.5, marker="o", c = '0.8', edgecolor='0.3', label = "Sim")

    df_copula.plot.scatter(x = df_copula.columns[2],
                           y = df_copula.columns[3],
                           ax = ax11,
                           label = 'Obs')
    ax11.legend()

    ax11.grid(color = '.7', linestyle='dotted', zorder=-1)
    ax11.set_xlabel(df_copula.columns[2], fontsize = 12)
    ax11.set_ylabel(df_copula.columns[3], fontsize = 12)

    # ------------ Original Domain
    ax12.scatter(df_sim.iloc[:, 0], df_sim.iloc[:, 1], s=20, alpha = 0.5, marker="o", c = '0.8', edgecolor='0.3', label = "Sim")
    df_copula.plot.scatter(x = df_copula.columns[0],
                           y = df_copula.columns[1],
                           ax = ax12,
                           label = 'Obs')
    ax12.legend(loc = 'lower right')

    # plt.text(0.85, 12, 'TC Matthew')
    tau, _ = kendalltau(df_copula.iloc[:, 0], df_copula.iloc[:, 1])
    plt.text(plt.xlim()[0], plt.ylim()[1], r"$\tau$" + f" = {tau:.2f}", fontsize=12, verticalalignment='top')
    plt.title(df_copula.columns[0] + ' vs ' + df_copula.columns[1])

    ax12.grid(color = '.7', linestyle='dotted', zorder=-1)
    ax12.set_xlabel(df_copula.columns[0], fontsize = 12)
    ax12.set_ylabel(df_copula.columns[1], fontsize = 12)

    plt.tight_layout()
    return None

def fit_vines(df_unity):
    cop_fam = [pv.BicopFamily(x) for x in np.arange(0,11)]
    controls = pv.FitControlsVinecop(
        family_set = cop_fam, 
        threshold = 0.05,
        selection_criterion = 'bic',
        # tree_criterion='tau',  # tau, hoeffd, rho, mcor -> no effect ?!
        show_trace=True,
    )
    cop = pv.Vinecop(df_unity.values, controls=controls)
    # print(cop)
    # print('----------')
    # print(cop.structure)
    # print('----------')
    return cop

def understand_vine(df_unity):
    cop = fit_vines(df_unity)
    data = []
    m = cop.matrix
    n = m.shape[0]
    drivers = df_unity.columns
    v = drivers
    for t in range(n-1):
        for e in range(n-1-t):
            p1, p2 = v[int(m[n-1-e,e]-1)], v[int(m[t,e]-1)]
            px = [v[int(p-1)] for p in m[:t,e]]
            c = cop.get_pair_copula(t,e)
            tau = cop.get_tau(t,e)  # NOTE: diffferent from scipy.stats method ?
            pxs = f' | ' + ','.join(px) if px else ''
            edge = f'{p1},{p2}{pxs}'
            cstr = c.str().replace(f'\n',',')
            print(f'{p1},{p2}{pxs}: {cstr}; tau = {tau:.5f}')
            data.append([t+1, e, edge, [p1,p2], px, c.str().split(',')[0], c.parameters.flatten(), tau])

    df_cop = pd.DataFrame(
        data=data,
        columns=['tree', 'edge#', 'edge', 'pair', 'conditional', 'copula', 'parameters', 'tau']
    )
    return df_cop, cop

def sim_vines(df_unity, n_sim, marginals, cop, columns, threshold = None, seeds = [5]):
    '''
    marginals is a list in the same order as df_unity

    last input is the seed, which is fixed, to have consistent simulations
    '''
    if threshold is None:
        threshold = len(marginals) * [[0, 1_000]]
    df_usim = pd.DataFrame(data=cop.simulate(n_sim, seeds = seeds), columns = df_unity.columns)
    for i in range(len(marginals)):
        curr_var = obtain_sim_values(df_usim.iloc[:, i], marginals[i], threshold[i])
        curr_var = curr_var.reshape(-1, 1)
        if i == 0:
            h_stack = curr_var
        else:
            h_stack = np.concatenate((h_stack, curr_var), axis = 1)
    df_sim = pd.DataFrame(h_stack, columns = columns)
    return df_usim, df_sim

def plot_sim_vine(df_sim, df_obs, df_cop, row_swap = None, new_ax = None, old_ax = None):
    plt.rc('legend',fontsize=15, title_fontsize=15)
    n = df_sim.shape[1]-1
    fig, axes = plt.subplots(n,n, sharex=False, sharey=False, figsize=(n*4,n*4), gridspec_kw={'hspace':0.0, 'wspace':0.0})

    for r in range(n):
        for c in range(n):
            if c > r:
                axes[r,c].set_visible(False)
                continue
    if new_ax is not None:
        axes[new_ax[0], new_ax[1]].set_visible(True)
        axes[old_ax[0], old_ax[1]].set_visible(False)

    for _, c0 in df_cop.iterrows():
        xlab, ylab = c0.pair
        c = c0['edge#'] 
        r = c + c0.tree - 1
        if row_swap is not None:
            dic_ind = list(row_swap.keys())
            if ylab == dic_ind[0]:
                r = row_swap[ylab]
            elif ylab == dic_ind[1]:
                if xlab != dic_ind[0]:
                    r = row_swap[ylab]
        if old_ax is not None:
            if r == old_ax[0] and c == old_ax[1]:
                r = new_ax[0]
                c = new_ax[1]
                xlab_long = find_column_long(df_sim, xlab)
                axes[r,c].set_xlabel(xlab_long)
        ax = axes[r,c]
        xlab_long = find_column_long(df_sim, xlab)
        ylab_long = find_column_long(df_sim, ylab)
        x, y = df_sim[xlab_long], df_sim[ylab_long]
        x_obs, y_obs = df_obs[xlab_long], df_obs[ylab_long]
        ax.scatter(x_obs, y_obs, s=20, label='Obs', zorder=2)
        ax.scatter(x, y, s=20, alpha = 0.2, marker="o", c = '0.8', edgecolor='0.3', label = "Sim")
        if r == n-1 and c == n-2:
            ax.legend(title='Sim Events', loc='upper left', bbox_to_anchor=(1.1, 3))
        kwargs = dict(        
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.5, lw=0.1)
        )
        cs = ' | ' + ','.join(c0.conditional) if c0.tree > 0 else ''
        ps = ','.join(c0.pair[::-1])
        ns = f'{c0.copula}' + r' ($\tau$=' + f'{c0.tau:.2f})' if c0.copula != 'Independence' else c0.copula
        txt = f'[{ps}{cs}]\n{ns}'
        ax.text(0.02, 0.98, txt, **kwargs)

        if c == 0:
            ax.set_ylabel(ylab_long)
        else:
            ax.set_yticklabels('')
        if r == n-1:
            ax.set_xlabel(xlab_long)
    return fig, axes
    
def find_column_long(df, short_name):
    long_name = df.columns[df.columns.str.contains(short_name)][0]
    return long_name