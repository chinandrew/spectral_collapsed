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
    # data_file = "testdata.p"
    configs = {
        # iters, thin, slab_size
        "mh_chol": [(25000, 1, float('inf'))],
        "inv_cdf_eig": [(25000, 1, float('inf'))],
        "unif_prior": [(150000, 6, float('inf'))],  # full uncollapsed conditional
    }
    config = configs[gscale_method]

    for n_iter, thin, slab_size in config:
        print(f"begin {gscale_method}", seed)
        with open('X_sparse_matrix_jhu.pickle', 'rb') as f:
            X = pickle.load(f) # 2035, 17909
        with open('y_vector_jhu.pickle', 'rb') as f:
            y = pickle.load(f)
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
            coef_sampler_type='woodbury',
            seed=seed, gscale_method=gscale_method, proposal_sd=proposal_sd
        )
        profiler.disable()
        print(sum(np.diff(samples["global_scale"])!=0))
        print(time.time() - start)
        print(pstats.Stats(profiler).total_tt)
        proposal_str = str(proposal_sd).replace(".", "-")
        pstats.Stats(profiler).dump_stats(f"{gscale_method}_{n_iter}_thin{thin}_slab{slab_size}_seed{seed}_proposal{proposal_str}_{socket.gethostname().split('.')[0]}.prof")
        with open(f"{gscale_method}_{n_iter}_thin{thin}_slab{slab_size}_seed{seed}_proposal{proposal_str}_{socket.gethostname().split('.')[0]}_samples.p", 'wb') as f:
            pickle.dump((samples, mcmc_info), f)

if __name__ == '__main__':
    sampler = sys.argv[1]
    print("seed", sys.argv[2])
    seed = int(sys.argv[2])
    try:
        proposal_sd = float(sys.argv[3])  # 0.25 achieves 44% acceptance rate
    except IndexError:
        proposal_sd = None
    main(sampler, seed, proposal_sd)
