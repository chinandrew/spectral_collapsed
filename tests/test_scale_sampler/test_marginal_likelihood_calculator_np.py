import numpy as np
import pytest
import scipy as sp

from bayesbridge.design_matrix.sparse_matrix import SparseDesignMatrix
from bayesbridge.design_matrix.dense_matrix import DenseDesignMatrix
from bayesbridge.global_scale_sampler.log_marginal_likelihood_calculator_np import LogMarginalLikelihoodCalculatorNP


@pytest.fixture
def X():
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.fixture
def obs_prec(X):
    return np.linspace(1, 2, X.shape[0])


@pytest.fixture
def y(X, obs_prec):
    return np.array([1 / 2, -1 / 2, -1 / 2]) / obs_prec


@pytest.fixture
def prior_coef_mean(X):
    return np.linspace(2, 3, X.shape[1])


@pytest.fixture
def prior_coef_var(X):
    return np.linspace(3, 4, X.shape[1])

@pytest.fixture
def prior_slab_size():
    return 1.5


def test_eval(y, obs_prec, X, prior_coef_mean, prior_coef_var):
    """Compare to naive calculation of marginal likelihood that does not use eigendecomp."""
    design = DenseDesignMatrix(X, add_intercept=True, center_predictor=False)
    prior_intrcpt_var = 1.4
    calc = LogMarginalLikelihoodCalculatorNP(y, design, np.sqrt(prior_intrcpt_var), np.sqrt(prior_coef_var),
                                           prior_coef_mean, obs_prec)
    gscale = 1.2
    y_centered = y - gscale * calc.outcome_mean_from_prior
    n = len(y)
    inv_marg_lik_cov = np.linalg.inv(
        np.diag(1 / obs_prec) + prior_intrcpt_var * np.outer(np.ones(n), np.ones(n)) + gscale ** 2 * X @ np.diag(prior_coef_var) @ X.T
    )
    log_det = -np.log(np.linalg.det(inv_marg_lik_cov))
    log_det -= np.log(prior_intrcpt_var)  # We ignore this in calculations
    exponent = -0.5 * y_centered @ inv_marg_lik_cov @ y_centered
    assert np.isclose(calc(gscale), -0.5 * log_det + exponent)


def test_eval_no_intrcpt(y, obs_prec, X, prior_coef_mean, prior_coef_var):
    """Compare to naive calculation of marginal likelihood w/o intercept that does not use eigendecomp."""
    design = DenseDesignMatrix(X, add_intercept=False, center_predictor=False)
    calc = LogMarginalLikelihoodCalculatorNP(y, design, None, np.sqrt(prior_coef_var),
                                           prior_coef_mean, obs_prec)
    gscale = 1.2
    y_centered = y - gscale * calc.outcome_mean_from_prior
    inv_marg_lik_cov_no_intrcpt = np.linalg.inv(np.diag(1 / obs_prec) + gscale ** 2 * X @ np.diag(prior_coef_var) @ X.T)
    log_det = -np.log(np.linalg.det(inv_marg_lik_cov_no_intrcpt))
    exponent = -0.5 * y_centered @ inv_marg_lik_cov_no_intrcpt @ y_centered
    assert np.isclose(calc(gscale), -0.5 * log_det + exponent)
