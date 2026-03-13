import numpy as np
from scipy.linalg import cho_factor, cho_solve


class LogMarginalLikelihoodCalculatorCholNP:

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
        self.y_obs_prec_y = (self.y * self.obs_prec) @ self.y
        self.y_obs_prec_X = design.Tdot(self.y * self.obs_prec)
        self.mu_obs_prec_X = design.Tdot(self.outcome_mean_from_prior * self.obs_prec)
        self.fisher_info_obs_prec = self.design.compute_fisher_info(self.obs_prec)

    def __call__(self, gscale):
        y_prior_centered =  self.y - gscale * self.outcome_mean_from_prior
        y_obs_prec_y = (y_prior_centered * self.obs_prec) @ y_prior_centered
        y_centered_obs_prec_X = self.y_obs_prec_X -  gscale * self.mu_obs_prec_X
        reg_prior_coef_var = self.prior_coef_var / (
                    1 + gscale ** 2 * self.prior_coef_var / self.prior_slab_size ** 2)
        if self.design.intercept_added:
            reg_prior_coef_var = np.concatenate(([self.prior_intrcpt_var], gscale**2*reg_prior_coef_var))
        else:
            reg_prior_coef_var *= gscale**2
        c, low = cho_factor(self.fisher_info_obs_prec + np.diag(1/reg_prior_coef_var))
        log_det = np.sum(np.log(np.diag(c))) * 2 + np.sum(np.log(1/self.obs_prec)) + np.sum(np.log(reg_prior_coef_var[self.design.intercept_added:]))  # We ignore the intercept variance, which can also be inf and cause issues,  since we only need proportionality
        exponent = y_obs_prec_y - y_centered_obs_prec_X @ cho_solve((c, low), y_centered_obs_prec_X)
        return -0.5 * log_det + -0.5 * exponent
