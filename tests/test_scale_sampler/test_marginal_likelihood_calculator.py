import numpy as np
import pytest
import scipy as sp

from bayesbridge.design_matrix.sparse_matrix import SparseDesignMatrix
from bayesbridge.global_scale_sampler.log_marginal_likelihood_calculator import LogMarginalLikelihoodCalculator


@pytest.fixture
def X():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def obs_prec(X):
    return np.linspace(1, 2, X.shape[0])


@pytest.fixture
def y(X, obs_prec):
    return np.array([1 / 2, -1 / 2]) / obs_prec


@pytest.fixture
def prior_coef_mean(X):
    return np.linspace(2, 3, X.shape[1])


@pytest.fixture
def prior_coef_var(X):
    return np.linspace(3, 4, X.shape[1])


def test_eval(y, obs_prec, X, prior_coef_mean, prior_coef_var):
    """Compare to naive calculation of marginal likelihood that does not use eigendecomp."""
    design = SparseDesignMatrix(sp.sparse.csr_matrix(X), add_intercept=True, center_predictor=False)
    prior_intrcpt_var = 1.4
    calc = LogMarginalLikelihoodCalculator(y, design, np.sqrt(prior_intrcpt_var), np.sqrt(prior_coef_var),
                                           prior_coef_mean, obs_prec)
    gscale = 1.2
    y_centered = y - gscale * calc.outcome_mean_from_prior
    n = len(y)
    inv_marg_lik_cov = np.linalg.inv(
        np.diag(1 / obs_prec) + prior_intrcpt_var * np.outer(np.ones(n), np.ones(n)) + gscale ** 2 * X @ np.diag(prior_coef_var) @ X.T
    )
    log_det = -np.log(np.linalg.det(inv_marg_lik_cov))
    log_det -= np.log(prior_intrcpt_var)  # unnormalize to align with what calc() returns
    exponent = -0.5 * y_centered @ inv_marg_lik_cov @ y_centered
    assert np.isclose(calc(gscale), -0.5 * log_det + exponent)


def test_eval_no_intrcpt(y, obs_prec, X, prior_coef_mean, prior_coef_var):
    """Compare to naive calculation of marginal likelihood w/o intercept that does not use eigendecomp."""
    design = SparseDesignMatrix(sp.sparse.csr_matrix(X), add_intercept=False, center_predictor=False)
    calc = LogMarginalLikelihoodCalculator(y, design, None, np.sqrt(prior_coef_var),
                                           prior_coef_mean, obs_prec)
    gscale = 1.2
    y_centered = y - gscale * calc.outcome_mean_from_prior
    inv_marg_lik_cov_no_intrcpt = np.linalg.inv(np.diag(1 / obs_prec) + gscale ** 2 * X @ np.diag(prior_coef_var) @ X.T)
    log_det = -np.log(np.linalg.det(inv_marg_lik_cov_no_intrcpt))
    exponent = -0.5 * y_centered @ inv_marg_lik_cov_no_intrcpt @ y_centered
    assert np.isclose(calc(gscale), -0.5 * log_det + exponent)


# @pytest.mark.parametrize("add_intercept, prior_intrcpt_var, expected",
#                          [(False, None, -0.2897204027322404),
#                           (True, np.inf, -0.7623808124424043),
#                           (True, 1.4, -0.3309069594323324)])
@pytest.mark.parametrize("add_intercept, prior_intrcpt_var, expected",
                         [(False, None, 0.9822258872967122),
                          (True, np.inf, 0.5116983724640187),
                          (True, 1.4, 0.9409198889407087)])
def test_ratio_inf_prior_intrcpt_var(y, obs_prec, X, prior_coef_mean, prior_coef_var, add_intercept, prior_intrcpt_var, expected):
    """Regression test to make sure ratio of likelihoods remains the same."""
    design = SparseDesignMatrix(sp.sparse.csr_matrix(X), add_intercept=add_intercept, center_predictor=False)
    calc = LogMarginalLikelihoodCalculator(y, design, prior_intrcpt_var, np.sqrt(prior_coef_var),
                                           prior_coef_mean, obs_prec)
    gscale1 = 1.2
    gscale2 = 2.1
    # print(calc(gscale1) - calc(gscale2))
    assert np.isclose(calc(gscale1) - calc(gscale2), expected)
