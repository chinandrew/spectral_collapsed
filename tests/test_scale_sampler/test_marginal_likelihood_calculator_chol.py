import numpy as np
import pytest
import scipy as sp

from bayesbridge.design_matrix.sparse_matrix import SparseDesignMatrix
from bayesbridge.global_scale_sampler.log_marginal_likelihood_calculator_chol import LogMarginalLikelihoodCalculatorChol


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

@pytest.fixture
def prior_slab_size():
    return 1.5

def test_eval(y, obs_prec, X, prior_coef_mean, prior_coef_var, prior_slab_size):
    """Compare to naive calculation of marginal likelihood that does not use eigendecomp."""
    design = SparseDesignMatrix(sp.sparse.csr_matrix(X), add_intercept=True, center_predictor=False)
    prior_intrcpt_var = 1.4
    gscale = 1.2
    regularized_prior_coef_var = prior_slab_size ** 2 * prior_coef_var / (
                prior_slab_size ** 2 + gscale ** 2 * prior_coef_var)
    calc = LogMarginalLikelihoodCalculatorChol(y, design, np.sqrt(prior_intrcpt_var), np.sqrt(prior_coef_var),
                                               prior_coef_mean, obs_prec, prior_slab_size)
    y_centered = y - gscale * calc.outcome_mean_from_prior
    n, p = X.shape
    inv_marg_lik_cov = np.linalg.inv(
        np.diag(1 / obs_prec) + prior_intrcpt_var * np.outer(np.ones(n), np.ones(n)) + gscale ** 2 * X @ np.diag(regularized_prior_coef_var) @ X.T
    )
    inv_marg_lik_cov2 = np.linalg.inv(
        np.diag(1 / obs_prec) + prior_intrcpt_var * np.outer(np.ones(n), np.ones(n)) +  X @ np.linalg.inv(np.diag(1/(gscale**2*prior_coef_var)) + np.eye(p)/prior_slab_size**2   ) @ X.T
    )
    assert np.allclose(inv_marg_lik_cov, inv_marg_lik_cov2)
    log_det = -np.log(np.linalg.det(inv_marg_lik_cov))
    log_det -= np.log(prior_intrcpt_var)  # unnormalize to align with what calc() returns
    exponent = -0.5 * y_centered @ inv_marg_lik_cov @ y_centered
    assert np.isclose(calc(gscale), -0.5 * log_det + exponent)


def test_eval_no_intrcpt(y, obs_prec, X, prior_coef_mean, prior_coef_var, prior_slab_size):
    """Compare to naive calculation of marginal likelihood w/o intercept that does not use eigendecomp."""
    design = SparseDesignMatrix(sp.sparse.csr_matrix(X), add_intercept=False, center_predictor=False)
    gscale = 1.2
    n, p = X.shape
    regularized_prior_coef_var = prior_slab_size ** 2 * prior_coef_var / (
                prior_slab_size ** 2 + gscale ** 2 * prior_coef_var)
    calc = LogMarginalLikelihoodCalculatorChol(y, design, None, np.sqrt(prior_coef_var),
                                               prior_coef_mean, obs_prec, prior_slab_size)
    y_centered = y - gscale * calc.outcome_mean_from_prior
    inv_marg_lik_cov_no_intrcpt = np.linalg.inv(np.diag(1 / obs_prec) + gscale ** 2 * X @ np.diag(regularized_prior_coef_var) @ X.T)
    inv_marg_lik_cov_no_intrcpt2 = np.linalg.inv(np.diag(1 / obs_prec) +  X @ np.linalg.inv(np.diag(1/(gscale**2*prior_coef_var)) + np.eye(p)/prior_slab_size**2) @ X.T)
    assert np.allclose(inv_marg_lik_cov_no_intrcpt, inv_marg_lik_cov_no_intrcpt2)
    log_det = -np.log(np.linalg.det(inv_marg_lik_cov_no_intrcpt))
    exponent = -0.5 * y_centered @ inv_marg_lik_cov_no_intrcpt @ y_centered
    assert np.isclose(calc(gscale), -0.5 * log_det + exponent)


@pytest.mark.parametrize("add_intercept, prior_intrcpt_var, expected",
                         [(False, None, 12.860203390573295),
                          (True, np.inf, 12.398220184277426),
                          (True, 1.4, 12.759603010918438)])
def test_ratio_inf_prior_intrcpt_var(y, obs_prec, X, prior_coef_mean, prior_coef_var, prior_slab_size, add_intercept, prior_intrcpt_var,  expected):
    """Regression test to make sure ratio of likelihoods remains the same."""
    design = SparseDesignMatrix(sp.sparse.csr_matrix(X), add_intercept=add_intercept, center_predictor=False)
    calc = LogMarginalLikelihoodCalculatorChol(y, design, prior_intrcpt_var, np.sqrt(prior_coef_var),
                                               prior_coef_mean, obs_prec, prior_slab_size)
    gscale1 = 1.2
    gscale2 = 2.1
    print(calc(gscale1) - calc(gscale2))
    assert np.isclose(calc(gscale1) - calc(gscale2), expected)
