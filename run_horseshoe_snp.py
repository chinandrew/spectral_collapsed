#!/usr/bin/env python
# coding: utf-8
import cProfile
import pickle
import pstats
import random
import socket
import numpy as np
import scipy as sp
import sys
import time

from bayesbridge import BayesBridge, RegressionModel
from bayesbridge import HorseshoePrior


def main(gscale_method, seed, proposal_sd):
    print(socket.gethostname())
    configs = {
        # iters, thin, slab_size
        "mh_chol": [(25000, 1, float('inf'))],
        "inv_cdf_eig": [(10000, 1, float('inf'))],
        "unif_prior": [(10000, 1, float('inf'))], # full uncollapsed conditional
        "reject_halfcauchy": [(10000, 1, float('inf'))],  # full uncollapsed conditional
    }
    config = configs[gscale_method]

    for n_iter, thin, slab_size in config:
        print(f"begin {gscale_method}", seed)
        with open("snp_Xy_binary_1379.p", "rb") as f:
            X,y = pickle.load(f)
            X = X.astype(float)
        print(X.shape)
        model = RegressionModel(
            y, X, family='logit',
            add_intercept=True, center_predictor=True,
        )
        prior = HorseshoePrior(
            sd_for_intercept=float('inf'),
            regularizing_slab_size=slab_size,
            skew_mean=0.,
            skew_sd=1.,
            global_scale_prior=None,
            centered=True,
            gscale_prior_dist="halfcauchy" if "halfcauchy" in gscale_method else "unif"
        )
        np.random.seed(seed)
        init = {"coef": np.zeros(X.shape[1]+1),
             "global_scale": 10**-(seed % 10),
             "local_scale": sp.stats.halfcauchy.rvs(size=X.shape[1])}
        print(init['global_scale'])
        bridge = BayesBridge(model, prior)
        random.seed(seed)
        np.random.seed(seed)
        start = time.time()
        profiler = cProfile.Profile()
        profiler.enable()
        samples, mcmc_info = bridge.gibbs(
            n_iter=n_iter, n_burnin=0, thin=thin,
            init=init,
            coef_sampler_type="cholesky",
            seed=seed, gscale_method=gscale_method, proposal_sd=proposal_sd,
            params_to_save=('coef', 'global_scale', 'logp', 'local_scale')
        )
        profiler.disable()
        print(sum(np.diff(samples["global_scale"])!=0))
        print(time.time() - start)
        print(pstats.Stats(profiler).total_tt)
        pstats.Stats(profiler).dump_stats(f"snp_{gscale_method}_{n_iter}_thin{thin}_slab{slab_size}_seed{seed}_{socket.gethostname().split('.')[0]}_proposal{str(proposal_sd).replace('.', '-')}_allparams.prof")
        with open(f"snp_{gscale_method}_{n_iter}_thin{thin}_slab{slab_size}_seed{seed}_{socket.gethostname().split('.')[0]}_proposal{str(proposal_sd).replace('.', '-')}_samples_allparams.p", 'wb') as f:
            pickle.dump((samples, mcmc_info), f)

if __name__ == '__main__':
    sampler = sys.argv[1]
    print("seed", sys.argv[2])
    seed = int(sys.argv[2])
    try:
        proposal_sd = float(sys.argv[3])  # 0.06 achieves 44% acceptance rate
    except IndexError:
        proposal_sd = None
    main(sampler, seed, proposal_sd)
