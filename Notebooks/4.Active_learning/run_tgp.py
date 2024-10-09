import numpy as np
import pandas as pd

from Notebooks.Scripts.samp_algo import mda_tgp

if __name__ == '__main__':

    if 'snakemake' in globals():
        snakemake = globals.get('snakemake')
        nvar = snakemake.params['nvar']
    else:
        nvar = "2d"

    class_list = pd.read_csv('fitted_stats/classified.csv', index_col=0).columns.to_list()
    sub_count_list = pd.read_csv('fitted_stats/sub_county.csv', index_col=0).columns.to_list()

    df_two = pd.read_csv(f'fitted_stats/{nvar}_sims.csv')

    folder_to_save = 'Notebooks/4.Active_learning/'

    mda_tgp(df_two, 'TGP_{nvar}_single', folder_to_save + '{nvar}_single/', total_samples = 100, restart = False)
    np.savetxt('fitted_stats/TGP_{nvar}_single.txt', [1])
