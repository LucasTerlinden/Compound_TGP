Sub Folder Contents
-----------------------

| Each sub folder represents a different iteration of the TGP
| on a different number of outputs or dimensions.
| single = complete model (1 output)
| multi = classifed model (2 outputs)
| sub = sub-county model (11 outputs)
| 
| acqui.csv: contains the alm statistics corresponding to the avilable samples (sample_space.csv)
| max_min.csv: contains the maximum and minimum consequences for that particular outputs
| plot_2.pdf: Figure containing locations of partitions (if they exist)
| plot.pdf: Figure containing TGP surface in first two dimensions (S and P Mag)
| sampled_events.csv: events that have been simulated along with output. Both are minmax normalized.
| stability.csv: Contains EAD output, alm statistic, 100 year RP and time of each iteration
| X*.csv: Statistics related to the surface output at sampled points (sampled_events.csv)
| XX*.csv: Statistics related to the surface output at available points (sample_space.csv)