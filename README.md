# CIDGIKc

A distance-based inverse kinematics solver for extensible segment constant curvature continuum robots.

## Environment setup

1. Install conda ([Miniconda3](https://docs.conda.io/en/latest/miniconda.html) recommended)
2. (Optional) install the [mamba](https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install) package. In this case, replace all the "conda" commands with "mamba" below.
3. Install MOSEK and obtain a license (free for academic use) https://www.mosek.com/downloads/
4. Run the following:

```sh
# clone the repo and navigate to the cidgikc directory
git clone https://github.com/ContinuumRoboticsLab/CIDGIKc.git
cd cidgikc

# Create & activate the conda env
conda env create -f environment.yml
conda activate cidgikc-env
```

5. Run the algorithm on a single problem instance by running the following:
 ```sh 
 cd scripts 
 python exp_sing_contrived.py
 ```
