# CIDGIKc

A distance-based inverse kinematics solver for extensible segment constant curvature continuum robots.

## Environment setup

1. Install conda ([Miniconda3](https://docs.conda.io/en/latest/miniconda.html) recommended)
2. (Optional) install the [mamba](https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install) package. In this case, replace all the "conda" commands with "mamba" below.
3. Install MOSEK and obtain a liscence (free for academic use) https://www.mosek.com/downloads/
4. Run the following:

```sh
git clone https://github.com/hanjzh/dgik4cr.git && cd dgik4cr

# Create & activate the conda env
conda env create -f environment.yml
conda activate dgik4cr-env

# Install the iPython kernel spec file (necessary to use the conda environment
# in Jupyter notebooks)
python -m ipykernel install --user --name dgik4cr-env
```
5. Run the algorithm on a single problem instance using "exp_sing_contrived.py" in the "scripts" directory.
