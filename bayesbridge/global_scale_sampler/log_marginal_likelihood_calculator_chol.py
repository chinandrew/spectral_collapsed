import numpy as np
from scipy.linalg import cho_factor, cho_solve


class LogMarginalLikelihoodCalculatorChol:

    def __init__(self, y, design, prior_intrcpt_sd, prior_shrunk_coef_sd, prior_shrunk_coef_mean, obs_prec,
                 prior_slab_size):
        self.y = y
        self.n_obs = len(y)
        self.design = design
        self.prior_coef_var = prior_shrunk_coef_sd ** 2
        self.obs_prec = obs_prec
        if design.intercept_added:
            self.prior_intrcpt_var = prior_intrcpt_sd ** 2
            prior_coef_mean = np.concatenate(([0], prior_shrunk_coef_mean))
        else:
            self.prior_intrcpt_var = 0
            prior_coef_mean = prior_shrunk_coef_mean
        self.outcome_mean_from_prior = design.dot(prior_coef_mean)
        self.obs_prec = obs_prec
        self.prior_slab_size = prior_slab_size

    def __call__(self, gscale):
        y_prior_centered = self.y - gscale * self.outcome_mean_from_prior
        reg_prior_coef_var = self.prior_coef_var / (1 + gscale ** 2 * self.prior_coef_var / self.prior_slab_size ** 2 )
        prec = np.diag(1 / self.obs_prec) + gscale ** 2 * self.design.compute_transposed_fisher_info(reg_prior_coef_var)
        c, low = cho_factor(prec)
        prec_inv_y = cho_solve((c, low), y_prior_centered)
        exponent = y_prior_centered @ prec_inv_y
        log_det = np.sum(np.log(np.diag(c))) * 2
        if self.design.intercept_added:
            prec_inv_one = cho_solve((c, low), np.ones(self.n_obs))
            one_prec_inv_one = np.ones(self.n_obs) @ prec_inv_one
            log_det += np.log(1 / self.prior_intrcpt_var + one_prec_inv_one)
            y_prec_inv_one = y_prior_centered @ prec_inv_one
            exponent -= y_prec_inv_one ** 2 / (1 / self.prior_intrcpt_var + one_prec_inv_one)
        return -0.5 * log_det + -0.5 * exponent
