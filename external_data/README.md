# Processed datasets

The small processed files in this directory are the inputs used by the
reproducible runner.  Each `.npy` file stores a dictionary of pandas data
frames with `Cycle` and `Capacity` columns (Stanford additionally contains the
processed SOH variants).  The runner keeps the held-out cell and starting
points in each experiment command/configuration.

The files are included in this private research repository so a fresh clone
can reproduce the local experiments.  Please preserve the original dataset
citations and licenses when redistributing the repository.

