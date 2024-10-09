#%%
import pandas as pd
df = pd.read_csv('all_drivers.csv', index_col=0, parse_dates=True)
#%%
df.round(3).to_csv('all_drivers.csv.gz', compression='gzip')

#%%
df1 = pd.read_csv('all_drivers.csv.gz', index_col=0, parse_dates=True)
df1['WL (MSL,m)']