# A Fully Bayesian Gradient-Free Supervised Dimension Reduction Method using Gaussian Processes

Github repository accompanying the paper entitled "A Fully Bayesian Gradient-Free Supervised Dimension Reduction Method using Gaussian Processes".

## Setup

The setup instructions assume you are using anaconda/miniconda.

 - clone this repository and `cd` to the project root
 - create a conda environment using the provided `environment.yml` file (`conda env create --file=environment.yml`)
 - activate this environment (its default name is `bayesian_dimension_reduction`)
 - install this project as a library (`pip install -e .`)
 - finally, clone and install [this fork of pymanopt](https://github.com/rhgautier/pymanopt)

## Run the code

The script `run_one_case.py` is the main point of entry. It features a set of parameters that can be modified to run a particular case. Running this script produces an `hdf5` file in a subfolder of the `results` directory that contains all relevant training and validation artifacts. `hdf5` files may be programatically opened, e.g. using [h5py](https://www.h5py.org/) or browsed using a utility such as [HDFView](https://www.hdfgroup.org/downloads/hdfview/).