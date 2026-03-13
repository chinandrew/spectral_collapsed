import numpy as np

from .log_marginal_likelihood_calculator import LogMarginalLikelihoodCalculator
from .log_marginal_likelihood_calculator_np import LogMarginalLikelihoodCalculatorNP
from .log_marginal_likelihood_calculator_chol import LogMarginalLikelihoodCalculatorChol
from .log_marginal_likelihood_calculator_chol_np import LogMarginalLikelihoodCalculatorCholNP
from .inverse_cdf_sampler import InverseCDFSampler


def coef_collapsed_sampler(gscale, obs_prec, y_gaussian, design, lscale, prior, rg, method="inv_cdf", **kwargs):
    """
    Sample global scale parameter `gscale`.

    Uses either Metropolis Hastings ("mh") or inverse CDF method ("inv_cdf").

    Samples on a log scale but returns on original scale.

    Parameters
    ----------
    gscale : float
        Current global scale parameter value.
    obs_prec : np.ndarray
        Length n vector of precision of y variables.
    y_gaussian : np.ndarray
        Length n vector of responses in where y ~ N(design.dot(coef), obs_prec ** -1)
    design : AbstractDesignMatrix
         Design matrix object holding n x p design matrix.
    lscale : np.ndarray
         Length p vector of local scale parameters.
    prior : HorseshoePrior
        HorseshoePrior class to evaluate prior density and mean/var for `gscale`.
    rg : BasicRandom
        BasicRandom object for generating random proposals.
    **kwargs
        Other arguments passed to the samplers. Currently would just be proposal sd for MH.

    Returns
    -------
    tuple[float, bool]
        (Next gscale value, True if proposal was accepted else False).
    """
    n, p = design.X_main.shape
    p += design.intercept_added
    if prior.slab_size < np.inf or "chol" in method:
        if n < p:
            log_marginal_likelihood = LogMarginalLikelihoodCalculatorChol(y_gaussian,
                                                                          design,
                                                                          prior.sd_for_intercept,
                                                                          lscale * prior.skew_sd,
                                                                          np.ones_like(lscale) * prior.skew_mean,
                                                                          obs_prec,
                                                                          prior.slab_size)
        else:
            log_marginal_likelihood = LogMarginalLikelihoodCalculatorCholNP(y_gaussian,
                                                                          design,
                                                                          prior.sd_for_intercept,
                                                                          lscale * prior.skew_sd,
                                                                          np.ones_like(lscale) * prior.skew_mean,
                                                                          obs_prec,
                                                                          prior.slab_size)
    else:
        if n < p:
            log_marginal_likelihood = LogMarginalLikelihoodCalculator(y_gaussian,
                                                                      design,
                                                                      prior.sd_for_intercept,
                                                                      lscale * prior.skew_sd,
                                                                      np.ones_like(lscale) * prior.skew_mean,
                                                                      obs_prec)
        else:
            log_marginal_likelihood = LogMarginalLikelihoodCalculatorNP(y_gaussian,
                                                                      design,
                                                                      prior.sd_for_intercept,
                                                                      lscale * prior.skew_sd,
                                                                      np.ones_like(lscale) * prior.skew_mean,
                                                                      obs_prec)
    if "mh" in method:
        return take_metropolis_step(gscale, lscale * prior.skew_sd, prior, log_marginal_likelihood, rg, **kwargs)
    elif "inv_cdf" in method:
        return inverse_cdf_sample(gscale, lscale * prior.skew_sd, prior, log_marginal_likelihood, rg)
    else:
        raise NotImplementedError


def inverse_cdf_sample(gscale, lscale, prior, log_marginal_likelihood, rg):
    logpdf_of_log_gscale = lambda x: log_marginal_likelihood(np.exp(x)) + np.log(prior.global_scale_prior(np.exp(x))) + x + np.sum(-0.5*np.log(1+np.exp(x)**2* lscale**2/prior.slab_size**2))
    sampler = InverseCDFSampler(logpdf_of_log_gscale, np.log(gscale), rg)
    return np.exp(sampler.sample())


def take_metropolis_step(gscale, lscale, prior, log_marginal_likelihood, rg, proposal_sd=1):
    proposal = np.exp(np.log(gscale) + rg.np_random.normal(0, proposal_sd))
    current_logdensity = np.log(prior.global_scale_prior(gscale)) + log_marginal_likelihood(gscale) + np.sum(-0.5*np.log(1+gscale**2 * lscale**2/prior.slab_size**2))
    proposal_logdensity = np.log(prior.global_scale_prior(proposal)) + log_marginal_likelihood(proposal) + np.sum(-0.5*np.log(1+proposal**2 * lscale**2/prior.slab_size**2))
    accept_prob_jacobian = np.log(proposal) - np.log(gscale)
    accept_prob = min(1, np.exp(proposal_logdensity - current_logdensity + accept_prob_jacobian))
    return proposal if rg.np_random.uniform() < accept_prob else gscale
