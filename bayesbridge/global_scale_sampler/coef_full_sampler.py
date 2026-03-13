import numpy as np
import scipy as sp


def coef_full_sampler(gscale, shape, rate, method):
    if method == "reject_halfcauchy":
        return rejection_sample_halfcauchy_prior(shape, rate)
    elif method == "slice_halfcauchy":
        return slice_sample_halfcauchy_prior(gscale, shape, rate)
    elif method == "unif_prior":
        return sample_unif_prior(shape-1, rate)
    else:
        raise NotImplementedError

def sample_unif_prior(shape, rate):
    return 1 / np.sqrt(truncated_gamma_lower(shape, rate, 1))

def rejection_sample_halfcauchy_prior(shape, rate):
    shape_offset = 1  # improve acceptance rates
    proposal = np.random.gamma(shape-shape_offset, 1 / rate)
    tries = 0
    while np.random.uniform() > proposal**shape_offset / (1 + proposal):
        proposal = np.random.gamma(shape-shape_offset, 1 / rate)
        tries += 1
    return 1 / np.sqrt(proposal)

def slice_sample_halfcauchy_prior(gscale, shape, rate):
    eta = 1 / gscale ** 2
    u = np.random.uniform(0, 1 / (1 + eta))
    return 1 / np.sqrt(truncated_gamma_upper(shape, rate, (1 - u) / u))


def truncated_gamma_upper(shape, rate, upper):
    gamma = sp.stats.gamma(a=shape, scale=1 / rate)
    u = np.random.uniform()
    return gamma.ppf(u * gamma.cdf(upper))

def truncated_gamma_lower(shape, rate, lower):
    gamma = sp.stats.gamma(a=shape, scale=1 / rate)
    lower_cdf = gamma.cdf(lower)
    u = np.random.uniform()
    return gamma.ppf(u * (1-lower_cdf) + lower_cdf)

