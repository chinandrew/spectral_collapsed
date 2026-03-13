import numpy as np


class LogMarginalLikelihoodCalculator:

    def __init__(self, y, design, prior_intrcpt_sd, prior_shrunk_coef_sd, prior_shrunk_coef_mean, obs_prec):
        self.y = y
        self.n_obs = len(y)
        self.design = design
        self.prior_coef_var = prior_shrunk_coef_sd ** 2
        if design.intercept_added:
            self.prior_intrcpt_var = prior_intrcpt_sd ** 2
            prior_coef_mean = np.concatenate(([0], prior_shrunk_coef_mean))
        else:
            self.prior_intrcpt_var = 0
            prior_coef_mean = prior_shrunk_coef_mean
        self.outcome_mean_from_prior = design.dot(prior_coef_mean)
        self.obs_prec = obs_prec
        obs_prec_sqrt = self.obs_prec ** 0.5
        self.tf_eigval, self.tf_eigvec = np.linalg.eigh(
            obs_prec_sqrt[np.newaxis, :]
            * self.design.compute_transposed_fisher_info(self.prior_coef_var)
            * obs_prec_sqrt[:, np.newaxis])
        self.tf_eigval = np.maximum(0, self.tf_eigval)
        self.obs_prec_sqrt_eigvec = obs_prec_sqrt[:, np.newaxis] * self.tf_eigvec
        self.obs_prec_sqrt_eigvec_y = self.obs_prec_sqrt_eigvec.T @ self.y
        self.y_obs_prec_y = (self.y * self.obs_prec) @ self.y
        self.obs_prec_sum = sum(obs_prec)
        self.obs_prec_sqrt_eigvec_1 = (obs_prec_sqrt[:, np.newaxis] * self.tf_eigvec).T @ np.ones(self.n_obs)
        self.y_obs_prec_1 = (self.y * self.obs_prec) @ np.ones(self.n_obs)
        self.y_obs_prec_mean = (self.y * self.obs_prec) @ self.outcome_mean_from_prior
        self.obs_prec_sqrt_eigvec_mean = (obs_prec_sqrt[:,
                                          np.newaxis] * self.tf_eigvec).T @ self.outcome_mean_from_prior
        self.mean_obs_prec_mean = (self.outcome_mean_from_prior * self.obs_prec) @ self.outcome_mean_from_prior
        self.mean_obs_prec_1 = (self.outcome_mean_from_prior * self.obs_prec) @ np.ones(self.n_obs)

    def __call__(self, gscale):
        woodburry_inv = (1 / self.tf_eigval + gscale ** 2) ** -1
        y_Ainv_y = self.y_obs_prec_y - gscale ** 2 * self.obs_prec_sqrt_eigvec_y.T @ (
                    woodburry_inv * self.obs_prec_sqrt_eigvec_y)
        output = y_Ainv_y
        if not np.allclose(self.outcome_mean_from_prior, 0):
            output -= 2 * gscale * (self.y_obs_prec_mean - gscale ** 2 * self.obs_prec_sqrt_eigvec_y.T @ (
                        woodburry_inv * self.obs_prec_sqrt_eigvec_mean))
            output += gscale ** 2 * (self.mean_obs_prec_mean - gscale ** 2 * self.obs_prec_sqrt_eigvec_mean.T @ (
                        woodburry_inv * self.obs_prec_sqrt_eigvec_mean))
        intrcpt_quad_form = None
        if self.design.intercept_added:
            intrcpt_quad_form = self.obs_prec_sum - gscale ** 2 * self.obs_prec_sqrt_eigvec_1.T @ (
                        woodburry_inv * self.obs_prec_sqrt_eigvec_1)
            y_Ainv_1 = self.y_obs_prec_1 - gscale ** 2 * self.obs_prec_sqrt_eigvec_y.T @ (
                        woodburry_inv * self.obs_prec_sqrt_eigvec_1)
            if not np.allclose(self.outcome_mean_from_prior, 0):
                mean_Ainv_1 = self.mean_obs_prec_1 - gscale ** 2 * self.obs_prec_sqrt_eigvec_mean.T @ (
                            woodburry_inv * self.obs_prec_sqrt_eigvec_1)
                output -= (y_Ainv_1 ** 2 - 2 * gscale * (y_Ainv_1 * mean_Ainv_1) + gscale ** 2 * mean_Ainv_1 ** 2) / (
                        1 / self.prior_intrcpt_var + intrcpt_quad_form)
            else:
                output -= (y_Ainv_1 ** 2) / (1 / self.prior_intrcpt_var + intrcpt_quad_form)
        exponent = output
        log_det = self.calc_marg_lik_cov_log_det(gscale, intrcpt_quad_form)
        return -0.5 * log_det + -0.5 * exponent

    def calc_marg_lik_cov_log_det(self, gscale, intrcpt_quad_form=None):
        """Calculate log determinant of the marginal likelihood covariance up to a factor of prior_intrcpt_var."""
        log_det = np.sum(np.log(1 + gscale ** 2 * self.tf_eigval)) - np.sum(np.log(self.obs_prec))
        if intrcpt_quad_form is not None:
            log_det += np.log(1 / self.prior_intrcpt_var + intrcpt_quad_form)
        return log_det
