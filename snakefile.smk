import pandas as pd

nvars = ["2d", "3d", "4d", "5d", "6d"]
counties = list(pd.read_csv("fitted_stats/sub_county.csv").columns)[1:]

rule all:
    input:
        "Data/skew_surge_tides.csv",
        "fitted_stats/precipitation.json",
        "fitted_stats/5d_sims.csv",
        "fitted_stats/6d_all_run.txt",
        "fitted_stats/TGP_6d_multi_run.txt",
        "fitted_stats/MDA_2d_run.txt",
        "fitted_stats/rerun_run.txt",
        "Figures/PNG/f03.png",
        "Figures/PNG/f04.png",
        "Figures/PNG/f11.png",


rule collect_all_drivers:
    input:
        "Data/raw_data/u10_era5.csv",
        "Data/raw_data/v10_era5.csv",
        "Data/raw_data/msl_era5.csv",
        "Data/raw_data/tp_era5.csv",
        "Data/raw_data/Charleston_meteo.csv",
        "Data/raw_data/Charleston_waterlevel.csv",
        "Data/raw_data/USGS.csv",
    output:
        "Data/all_drivers.csv",
        "Data/skew_surge_tides.csv",
    notebook:
        "Notebooks/1.Process_data/collect_all_drivers.ipynb"

# identifying extreme events & fitting extreme value distributions
rule extreme_value_analysis:
    input:
        "Data/all_drivers.csv",
        "Data/skew_surge_tides.csv",
    output:
        "fitted_stats/precipitation.json",
        "fitted_stats/skew_surge.json",
        "fitted_stats/compound_given_ss.csv",
        "fitted_stats/extreme_rate.txt",
    notebook:
        "Notebooks/2.Statistics/1.Identify_extremes.ipynb"


# fitting vine and generating events
rule vine_copula:
    input:
        "Data/all_drivers.csv",
        "Data/skew_surge_tides.csv",
        "fitted_stats/precipitation.json",
        "fitted_stats/skew_surge.json",
        "fitted_stats/compound_given_ss.csv",
    output:
        "fitted_stats/historical_eventset.csv",
        "fitted_stats/training_2d.csv",
        "fitted_stats/training_6d.csv",
        expand("fitted_stats/{nvar}_sims.csv", nvar=nvars),
        "fitted_stats/emperical_tide.csv",
        "fitted_stats/precipitation_dur.json",
        "fitted_stats/precipitation_lag.json",
        "fitted_stats/surge_dur.json",
    notebook:
        "Notebooks/2.Statistics/2.joint_prob.ipynb"

# run benchmark of 500 synthethic events in 2 and 6 dimensions
rule run_benchmark:
    input:
        "fitted_stats/training_2d.csv",
        "fitted_stats/training_6d.csv",
        # all sfincs basemodel files
        "Models/FloodAdapt_stolen/sfincs.inp",
        # all fiat basemodel files
        "Models/fiat_model/settings.toml",
    output:
        "fitted_stats/2d_all_run.txt",
        "fitted_stats/6d_all_run.txt",
        "Models/2d_all/damages.csv",
        "Models/6d_all/damages.csv",
    notebook:
        "Notebooks/3.Benchmark/obtain_benchmark.ipynb"

# surrogate & risk model
rule run_tgp:
    input:
        "fitted_stats/2d_all_run.txt",
        "fitted_stats/6d_all_run.txt",
        # classification of buildings for subcounties and coastal/inland
        "fitted_stats/fiat_indkeep.csv",
        "fitted_stats/sub_county.csv",
        "fitted_stats/classified.csv",
        # required for risk analysis
        "fitted_stats/extreme_rate.txt",
        "fitted_stats/historical_eventset.csv", # needed to define constants if d < 6
        # all outputs of the vine copula
        expand("fitted_stats/{nvar}_sims.csv", nvar=nvars),
        # all sfincs basemodel files
        "Models/FloodAdapt_stolen/sfincs.inp",
        # all fiat basemodel files
        "Models/fiat_model/settings.toml",
    # params:
    #     nvar="{nvar}"
    output:
        "Notebooks/4.Active_learning/2d_single/max_min.csv",
        # model run times
        expand("Models/TGP_{nvar}_multi/times.csv", nvar=nvars),
        expand("Models/TGP_{nvar}_single/times.csv", nvar=nvars),
        expand("Models/TGP_2d_sub/times.csv", nvar=nvars), # TODO make separate rule if script
        # EAD estimates per n simulations
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/stability.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/coast/stability.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/inland/stability.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/2d_sub/{county}/stability.csv", county=counties), 
        # TGP sampled events
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/sampled_events.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/coast/sampled_events.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/inland/sampled_events.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/2d_sub/{county}/sampled_events.csv", county=counties),
        # TGP last estimate for all samples for one output
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/X_mean.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/XX_mean.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/X_5.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/XX_5.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/X_95.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/XX_95.csv", nvar=nvars),
        # Check all loops are closed
        expand("fitted_stats/TGP_{nvar}_single_run.txt", nvar=nvars),
        expand("fitted_stats/TGP_{nvar}_multi_run.txt", nvar=nvars),
        "fitted_stats/TGP_2d_sub_run.txt", 
    notebook:
        "Notebooks/4.Active_learning/all_TGP.ipynb"

# run state of the art in 2 dimensions
rule run_state_art:
    input:
        "fitted_stats/2d_all_run.txt",
        "fitted_stats/TGP_2d_single_run.txt",
        "fitted_stats/2d_sims.csv",
        # all sfincs basemodel files
        "Models/FloodAdapt_stolen/sfincs.inp",
        # all fiat basemodel files
        "Models/fiat_model/settings.toml",
    output:
        "Models/MDA_2d/damages.csv",
        "Models/MDA_2d/times.csv",
        "fitted_stats/MDA_2d_run.txt",
    notebook:
        "Notebooks/5.Experiment/1a.State_art/mda_grid.ipynb" 

# Obtain TGP estimates on benchmark using the events sampled in run_tgp rule
rule rerun_TGP:
    input:
        "Notebooks/4.Active_learning/2d_single/Total/sampled_events.csv",
        "fitted_stats/2d_sims.csv",
        "fitted_stats/training_2d.csv",
        "fitted_stats/TGP_2d_single_run.txt",
        "fitted_stats/MDA_2d_run.txt",
    output:
        "Notebooks/5.Experiment/1b.TGP_accuracy/stats.csv",
        "fitted_stats/rerun_run.txt",
    notebook:
        "Notebooks/5.Experiment/1b.TGP_accuracy/rerun_TGP.ipynb" 

# Compare risk estimates, accuracy, and timing of both approaches
rule validate_TGP:
    input:
        "fitted_stats/2d_sims.csv",
        "fitted_stats/training_2d.csv",
        # benchmark damages
        "Models/2d_all/damages.csv",
        # MDA SFINCS + FIAT damages and times from 64 simulations
        "Models/MDA_2d/times.csv",
        "Models/MDA_2d/damages.csv",
        # TGP scaler, events, times
        "Notebooks/4.Active_learning/2d_single/max_min.csv",
        "Notebooks/4.Active_learning/2d_single/Total/sampled_events.csv",
        "Notebooks/4.Active_learning/2d_single/Total/stability.csv",
        # TGP SFINCS, FIAT times
        "Models/TGP_2d_single/times.csv",
        # ensure all loops are closed
        "fitted_stats/MDA_2d_run.txt",
        "fitted_stats/rerun_run.txt",
        "fitted_stats/TGP_2d_single_run.txt",
    output:
        "Figures/PNG/f08.png",
        "Figures/PNG/f02.png",
        "Figures/PNG/f03.png",
    notebook:
        "Notebooks/5.Experiment/1c.Compare/accuracy_times.ipynb"  

# Plot timing of each TGP model
rule scalability_TGP:
    input:
        "fitted_stats/classified.csv",
        "fitted_stats/sub_county.csv",
        # MDA SFINCS + FIAT times
        "Models/MDA_2d/times.csv",
        # TGP times per n simulations
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/stability.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/coast/stability.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/inland/stability.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/2d_sub/{county}/stability.csv", county=counties), 
        # TGP sampled events
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/sampled_events.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/coast/sampled_events.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_multi/inland/sampled_events.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/2d_sub/{county}/sampled_events.csv", county=counties),
        # SFINCS and FIAT times for TGP
        expand("Models/TGP_{nvar}_multi/times.csv", nvar=nvars),
        expand("Models/TGP_{nvar}_single/times.csv", nvar=nvars),
        expand("Models/TGP_2d_sub/times.csv", nvar=nvars),
        # Ensure all loops are closed for TGP and MDA
        expand("fitted_stats/TGP_{nvar}_single_run.txt", nvar=nvars),
        expand("fitted_stats/TGP_{nvar}_multi_run.txt", nvar=nvars),
        "fitted_stats/TGP_2d_sub_run.txt", 
        "fitted_stats/MDA_2d_run.txt",
    output:
        "Figures/PNG/f04.png",
    notebook:
        "Notebooks/5.Experiment/2.Scalability/timing.ipynb" 

# Plot risk estimates of each TGP model
rule risk_TGP:
    input:
        "fitted_stats/extreme_rate.txt",
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/sampled_events.csv", nvar=nvars),
        expand("fitted_stats/TGP_{nvar}_single_run.txt", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/X_mean.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/XX_mean.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/X_5.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/XX_5.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/X_95.csv", nvar=nvars),
        expand("Notebooks/4.Active_learning/{nvar}_single/Total/XX_95.csv", nvar=nvars),
    output:
        "Figures/PNG/f05.png",
        "Figures/PNG/f10.png",
        "Figures/PNG/f11.png",
    notebook:
        "Notebooks/5.Experiment/3.Risk_estimates/sensitivity_dim.ipynb" 