| This folder contains sub folders for the different dimensionalities or outputs. Each fodler contains: SFINCS and FIAT folders, damage and times csv. 

Models folder outline
-----------------------

::

  > *d_all
    These folders were created for the training dataset. The * denotes the dimensionality.
  > fiat_model
    Fiat model used in FloodAdapt. Everytime an event is simulated, it will duplicate this folder and overwrite the hazard map.
  > FloodAdapt_stolen
    SFINCS model used in FloodAdapt. Everytime an event is simulated, it will use the content of this folder and add the boundary condtion information from the event.
  > MDA_2d
    Simulations for the current approach in 2 dimensions
  > Placeholder
    Folder used to create the boundary condition schematization (Fig. A1) in the Notebooks/Additional folder
  > Sensitivity_mags
    Simulations used to conduct the preliminary sensitivity analysis
  > TGP_*
    All the simulations related to the Treed Gaussian Process. <sub> = sub-county model, <multi> = classified model, <single> = complete model.