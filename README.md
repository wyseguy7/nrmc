
This is the readme explaining the codebase for the paper [Herschlag et. al 2020](https://arxiv.org/abs/2008.07843).

## Installation and Running

Installation is accomplished by running:

`python setup.py build` followed by  `python setup.py install`.
By default, Windows systems will not attempt to install the Cython speedups due to difficulties with the compiler - if these are desired, it is necessary to modify the `setup.py` script.
Running the codebase will also require the `pandas`, `networkx`, `numpy`, and ``Cython` libraries.

## Sampling Runs

Sampling runs can be executed via the `run.py` script. An example run can be launched here:

`python src/scripts/run.py --steps 10000000  --process center_of_mass  --output_path /gtmp/etw16/runs/ --num_districts 3 --score_func cut_length --score_weights 1.0 --apd 0.1 --n 40`
The above script will launch a 10-million sample run using center-of-mass proposals over a 40 x 40 lattice.

The script will dump a copy of the state at intervals of 1 million iterations, and again once the run has completed.
The data is encoded both in a `pkl` and `json` format, though the use of the `pkl` format is more fully-featured


### Tracking Completed Runs

The `parameter_identifier.py` script can be used to generate or update a `csv` file with all of the completed runs within a given folder.
This is both useful for tracking and used as an input for other scripts below.
In order to ensure that all files are detected, it may be necessary to modify the input to the `glob.glob` command to account for different folder structures.

### Computing ranked-order marginals

The ranked-order marginals used in the paper are computed using the `calculate_marginal.py` script.
An example call to this script would be:
`python calculate_marginal.py --filepaths features_out.csv --vote_file data/Mecklenburg/Mecklenburg_P__EL12G_PR.txt`
This will compute the marginals for each file referenced within `features_out.csv`, and store it as a `marginal.csv` file within the relevant folder for each run.
Note that the `features_out.csv` file referenced here ought to be formatted like the output of the `parameter_identifier.py` script.


### Computing TV distance for an observeable

This can be accomplished using the `tv_dist_new.py` script.
It will expect to find a file named `marginals.csv` for each folder referenced in the input CSV, though this is easily modified.
The output will include an estimate of the TV distance at regular intervals.


### Computing G(t)

The G(t) function mentioned in the paper can be computed by running the `autocorr_calculator.py` script. An example:

`python autocorr_calculator.py --filepaths features_out.csv --max_distance 5000000`
