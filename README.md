# Code for "Spectral Collapsed Gibbs Sampler for Bayesian Sparse Regression"

Derived from the [bayes-bridge](https://github.com/OHDSI/bayes-bridge) package.

After installing (using, e.g. `pip install .` from the root directory), code to run simulations for Section 4.1 is `run_horseshoe_snp.py`, and code for 4.2 is `run_horseshoe_ehr.py`.
`run_horseshoe_snp.sh` contains an example shell script for executing `run_horseshoe_snp.py` via SLURM.

Data for Section 4.1 can be generated via the commands described in `snp_data.md` and is available pickled as `snp_Xy_binary_1379.p` via git LFS. Data for Section 4.2 is protected PHI and not publically available. 

