| Compound_TGP.
| Author: Lucas Terlinden-Ruhl.
| Contact: lucas.terlindenruhl@gmail.com.
| 
| Repository created to store data, scripts, and notebooks related to the research paper titled:
| Accelerating compound flood risk assessments through active learning: A case study of Charleston County (USA)

How to install & run
--------------------
:: 

  > conda env create -f environment.yml
  > conda env create -f environment_2.yml
  > conda activate run_compound
  > pip install git+https://github.com/Deltares/Delft-FIAT.git
  > pip install git+https://github.com/theochem/Selector.git
| R should also be installed, and added to the path of your system. The <tgp> package should also be installed seperately in R.

Repository outline
------------------

::

  > Data
    Contains raw data required for analysis. Run the notebook in 1.Process_data to post process
  > Executables
    Does not contain SFINCS or FIAT executables. These need to be obtained from Deltares.
    The folders containing the executables should be named:
    "SFINCS_executable" and "FIAT_executable"
  > Figures
    Will save all figures (apart from figure 1 and 2) here in png and 300 dpi pdf.
  > fitted_stats
    Contains files that were saved during the analysis. Currently only contains: sub-county,
    classified, and FIAT definitions. During the statistics step, additional files will be saved:
    historical event set, stochastic event sets, probability distribution functions, and extreme rate.
  > Models
    SFINCS and DELFT-FIAT models are saved here. Make sure to have enough memory. To save memory, it is 
    recommended to only keep the following files: "damages.csv", and "times.csv".
  > Notebooks
    Any (post) processing occurs here. The folders are orderd in a logical step to step basis.
    > 1.Process_data
      Process the raw data
    > 2.Statistics
      Generate stochastic event sets using a statisitcal model that identified past extremes and fits (vine) copulas
    > 3.Benchmark
      Run the benchmark event sets to obtain damages for all sythetic events.
    > 4.Active_learning
      Run the TGP-LLM on all test event sets on 1 and 2 outputs. Run the TGP-LLM on the test event set in 6 dimensions.
      ! Seed is not set: will obtain different results to paper.
    > 5.Experiment
      Perform the 3 experiments outlined in the paper
      > 1a.State_art
        Run the equidistant sampling on the test event set in 2 dimensions, and use linear scatter interpolation
        to infer the damages of the benchamrk event set.
      > 1b.TGP_accuracy
        Use the synthetic events selected by the active learning algorithm in 4. and obtain estimates for the benchamrk
        in 2 dimensions.
      > 1c.Compare
        Obtain RMSE and computational time estimates for both approaches.
      > 2.Scalability
        Plot the computational time of the TGP-LLM for different number of simplifications, and compare with state-of-the-art.
      > 3.Risk_estimates
        Plot the risk estimates obtained from the TGP-LLM on 2, 3, 4, 5, and 6 dimensions. 
    > Additional
      Contains notebooks that support assumptions or produce appendix figures.
      > stop_crit
        Run the TGP-LLM on the benchamrk event set to define a stopping criterion.
    > Scripts
      Any .py file is stored here, a one liner description is given at the start to give context
  > Visio
    Contains the workflow shown in Figure 1.
