import numpy as np
import scipy as sp

def generate_gaussian_with_weight(design, obs_prec, prior_prec_sqrt, z, rand_gen=None):
    """
    Generate a multi-variate Gaussian with covariance Sigma
        Sigma^{-1} = X diag(obs_prec) X + diag(prior_prec_sqrt) ** 2
    and mean = Sigma z, where X is the `design` matrix.

    Parameters
    ----------
        obs_prec : 1-d numpy array
        prior_prec_sqrt : 1-d numpy array
    """

    diag = prior_prec_sqrt ** 2 \
           + design.compute_fisher_info(weight=obs_prec, diag_only=True)
    jacobi_precond_scale = 1 / np.sqrt(diag)
    Prec_precond = compute_precond_post_prec(
        design, obs_prec, prior_prec_sqrt, jacobi_precond_scale
    )
    Prec_precond_chol = sp.linalg.cholesky(Prec_precond, jacobi_precond_scale)
    mean_precond = sp.linalg.cho_solve(
        (Prec_precond_chol, False), jacobi_precond_scale * z
    )
    if rand_gen is None:
        gaussian_vec = np.random.randn(design.shape[1])
    else:
        gaussian_vec = rand_gen.np_random.randn(design.shape[1])
    sample_precond = mean_precond
    sample_precond += sp.linalg.solve_triangular(
        Prec_precond_chol, gaussian_vec, lower=False
    )
    sample = jacobi_precond_scale * sample_precond

    return sample

def compute_precond_post_prec(design, obs_prec, prior_prec_sqrt, precond_scale):
    Prec_precond = \
        precond_scale[:, np.newaxis] \
        * design.compute_fisher_info(obs_prec) \
        * precond_scale[np.newaxis, :]
    Prec_precond += np.diag((precond_scale * prior_prec_sqrt) ** 2)
    return Prec_precond

def generate_gaussian_via_woodbury(design, obs_prec, prior_prec_sqrt, z):
    """
    Sample from a multi-variate Gaussian using the identity
        (D + X' W X)^{-1}
            = D^{-1} - D^{-1} X' (W^{-1} + X D^{-1} X')^{-1} X D^{-1}.
    """
    if np.any(prior_prec_sqrt[1:] == 0):
        raise NotImplementedError(
            "Woodbury sampler currently does not support flat prior on fixed effects."
        )

    # Draw a "target" vector, right-hand side of the linear system to be solved.
    randn_vec_1 = np.random.randn(design.shape[0])
    randn_vec_2 = np.random.randn(design.shape[1])
    v = design.Tdot(obs_prec ** (1 / 2) * randn_vec_1) \
        + prior_prec_sqrt * randn_vec_2
    rhs_target_vec = (z + v)
    sample = matvec_by_post_prec_inverse_via_woodbury(
        design, obs_prec, prior_prec_sqrt, rhs_target_vec
    )
    return sample

def matvec_by_post_prec_inverse_via_woodbury(design, obs_prec, prior_prec_sqrt, x):
    block21 = design.main_Tdot(obs_prec)
    block11 = prior_prec_sqrt[0]**2 + np.sum(obs_prec)
    D_inv = prior_prec_sqrt[1:] ** -2
    intercept_term, x = x[0], x[1:]
    to_be_inverted = \
        np.diag(obs_prec ** - 1) \
        + design.compute_transposed_fisher_info(weight=D_inv, include_intrcpt=False)
    woodbury_solve1, chol = solve_via_chol(to_be_inverted, design.main_dot(D_inv * x))
    block22_inv_x =  D_inv * x - D_inv * design.main_Tdot(woodbury_solve1)
    woodbury_solve2, _ = solve_via_chol(to_be_inverted, design.main_dot(D_inv * block21), chol)
    block22_inv_block_21 =  D_inv * block21 - D_inv * design.main_Tdot(woodbury_solve2)
    schur_complement_inv = 1 / (block11 - block21.T @ block22_inv_block_21)
    output = np.empty(design.shape[1])
    output[0] = schur_complement_inv * intercept_term - schur_complement_inv * block21.T @ block22_inv_x
    output[1:] = -block22_inv_block_21 * schur_complement_inv * intercept_term + block22_inv_x + block22_inv_block_21 * schur_complement_inv * (block21.T @ block22_inv_x)
    return output

def solve_via_chol(pos_def_mat, x, precomputations=None):
    # Use Jacobi preconditioner to improve numerical stability.
    if precomputations is None:
        precond_scale = 1 / np.diag(pos_def_mat)
        precond_mat = \
            precond_scale[:, np.newaxis] \
            * pos_def_mat \
            * precond_scale[np.newaxis, :]
        chol = sp.linalg.cho_factor(precond_mat)
    else:
        chol, precond_scale = precomputations
    result = precond_scale * x
    result = sp.linalg.cho_solve(chol, result)
    result *= precond_scale
    return result, (chol, precond_scale)

