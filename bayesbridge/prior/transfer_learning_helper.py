# TODO: find a way to compute/store mu_beta_a and var_beta_a,
#  one option is to store them in a csv file,
#  then for each beta, you find whether it's informed or not
import numpy as np
from ..random.local_scale_sampler import skewed_shrinkage_rejection_sampler
from ..random.local_scale_sampler.helper import x_to_eta


def compute_horseshoe_lscale(beta_coef,
                             gscale,
                             skew_prior_mean,
                             skew_prior_sd,
                             q=0.5, k1=16, k2=5):
    a = np.abs(beta_coef) / (np.sqrt(2) * gscale * skew_prior_sd)
    c = gscale * skew_prior_mean / beta_coef

    a[a == 0] = 2. ** -100
    a[np.isinf(a)] = 2. ** 100

    c[np.isinf(c)] = np.sign(c[np.isinf(c)]) * 2. ** 100

    rv_ar_list = [
        skewed_shrinkage_rejection_sampler(a[i], c[i], q=q, k1=k1, k2=k2)
        for i in range(len(a))
    ]
    # print(rv_ar_list)
    rv = np.array([i[0] for i in rv_ar_list]).flatten()
    acc_count = np.array([i[1] for i in rv_ar_list]).flatten()

    eta = x_to_eta(rv)
    return 1 / eta, acc_count


def get_informed_prior(mu_beta_a, r, dist, sigma_sq_ave, sigma_sq_db):
    sigma_sq_coef = sigma_sq_ave + sigma_sq_db
    rho = np.exp(- r * dist)
    gamma = rho * sigma_sq_ave / sigma_sq_coef
    gamma_sq = gamma ** 2
    return gamma * mu_beta_a, np.sqrt((1 - gamma_sq) * sigma_sq_coef)
