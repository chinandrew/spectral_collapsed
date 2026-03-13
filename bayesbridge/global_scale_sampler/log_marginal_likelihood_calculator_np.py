import numpy as np


class LogMarginalLikelihoodCalculatorNP:

    def __init__(self, y, design, prior_intrcpt_sd, prior_shrunk_coef_sd, prior_shrunk_coef_mean, obs_prec):
        self.y = y
        self.n_obs = len(y)
        self.p = len(prior_shrunk_coef_sd)
        self.design = design
        self.prior_shrunk_coef_sd = prior_shrunk_coef_sd
        self.prior_coef_var = prior_shrunk_coef_sd ** 2
        if design.intercept_added:
            self.prior_intrcpt_var = prior_intrcpt_sd ** 2
            prior_coef_mean = np.concatenate(([0], prior_shrunk_coef_mean))
        else:
            self.prior_intrcpt_var = 0
            prior_coef_mean = prior_shrunk_coef_mean
        self.outcome_mean_from_prior = design.dot(prior_coef_mean)
        self.obs_prec = obs_prec
        self.tf_eigval, self.tf_eigvec = np.linalg.eigh(
            self.prior_shrunk_coef_sd[np.newaxis, :]
            * self.design.compute_fisher_info(obs_prec)[design.intercept_added:, design.intercept_added:]
            * self.prior_shrunk_coef_sd[:, np.newaxis])
        self.tf_eigval = np.maximum(0, self.tf_eigval)
        self.outcome_mean_from_prior = design.dot(prior_coef_mean)
        self.y_obs_prec_X_int = sum(self.y * self.obs_prec)
        self.mu_obs_prec_X_int = sum(self.outcome_mean_from_prior * self.obs_prec)

        self.obs_prec_sum = sum(obs_prec)
        self.inv_obs_prec_logdet = np.sum(np.log(1 / self.obs_prec))

        self.one_obs_prec_X_sd = design.Tdot(self.obs_prec)[design.intercept_added:] * self.prior_shrunk_coef_sd
        self.one_obs_prec_X_sd_eigvec = self.tf_eigvec.T @ self.one_obs_prec_X_sd
        self.one_obs_prec_X_sd_T_one = self.one_obs_prec_X_sd @ self.one_obs_prec_X_sd

        self.y_obs_prec_X_sd = design.Tdot(self.y * self.obs_prec)[design.intercept_added:] * self.prior_shrunk_coef_sd
        self.mu_obs_prec_X_sd = design.Tdot(self.outcome_mean_from_prior * self.obs_prec)[design.intercept_added:] * self.prior_shrunk_coef_sd
        self.y_obs_prec_X_sd_eigvec = self.tf_eigvec.T @ self.y_obs_prec_X_sd
        self.mu_obs_prec_X_sd_eigvec = self.tf_eigvec.T @ self.mu_obs_prec_X_sd

        self.y_obs_prec_X_sd_T_y = self.y_obs_prec_X_sd @ self.y_obs_prec_X_sd
        self.y_obs_prec_X_sd_T_one = self.y_obs_prec_X_sd @ self.one_obs_prec_X_sd

    def __call__(self, gscale):
        y_prior_centered = self.y - gscale * self.outcome_mean_from_prior
        y_obs_prec_y = (y_prior_centered * self.obs_prec) @ y_prior_centered
        y_centered_obs_prec_X_int = self.y_obs_prec_X_int - gscale * self.mu_obs_prec_X_int
        log_det = np.sum(np.log(1 + gscale ** 2 * self.tf_eigval)) + self.inv_obs_prec_logdet

        y_cent_obs_prec_X_sd =  self.y_obs_prec_X_sd - gscale * self.mu_obs_prec_X_sd
        y_cent_obs_prec_X_sd_eigvec =  self.y_obs_prec_X_sd_eigvec - gscale * self.mu_obs_prec_X_sd_eigvec
        if self.design.intercept_added:
            one_Dinv_one = self._quadratic_a_Dinv_b(gscale, self.one_obs_prec_X_sd_T_one, self.one_obs_prec_X_sd_eigvec, self.one_obs_prec_X_sd_eigvec)
            one_Dinv_y = self._quadratic_a_Dinv_b(gscale, y_cent_obs_prec_X_sd @ self.one_obs_prec_X_sd, self.one_obs_prec_X_sd_eigvec, y_cent_obs_prec_X_sd_eigvec)
            y_Dinv_y = self._quadratic_a_Dinv_b(gscale, y_cent_obs_prec_X_sd @ y_cent_obs_prec_X_sd, y_cent_obs_prec_X_sd_eigvec, y_cent_obs_prec_X_sd_eigvec)
            schur_inv = 1/(self.obs_prec_sum + self.prior_intrcpt_var ** -1 - one_Dinv_one)

            log_det += np.log(
                self.obs_prec_sum + self.prior_intrcpt_var ** -1 - one_Dinv_one
            )
            # log_det += np.log(self.prior_intrcpt_var) # We ignore this proportionality term, which can also be inf and cause issues
            exponent = y_obs_prec_y - (y_centered_obs_prec_X_int**2 * schur_inv - 2 * y_centered_obs_prec_X_int*schur_inv*one_Dinv_y + (y_Dinv_y + one_Dinv_y**2*schur_inv))
        else:
            exponent = y_obs_prec_y
            exponent -=  self._quadratic_a_Dinv_b(gscale, y_cent_obs_prec_X_sd @ y_cent_obs_prec_X_sd, y_cent_obs_prec_X_sd_eigvec, y_cent_obs_prec_X_sd_eigvec)
        return -0.5 * log_det + -0.5 * exponent


    def _quadratic_a_Dinv_b(self, gscale, aTb, a_eigvec, b_eigvec):
        return gscale**2 * (aTb - gscale ** 2 * a_eigvec * 1 / (1 / self.tf_eigval + gscale ** 2) @ b_eigvec)

