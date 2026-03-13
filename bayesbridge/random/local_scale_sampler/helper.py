import sys
# sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt


def log_prob_c_pos(x, a, c):
    return - a ** 2.0 * (np.sqrt(np.exp(2.0 * x) - 1.0) - c) ** 2.0


def log_prob_c_neg(x, a, c):
    eta = np.exp(2.0 * x) - 1.0
    return - a ** 2.0 * eta + 2.0 * a ** 2.0 * c * np.sqrt(eta)


def log_prob(x, a, c):
    if c >= 0.0:
        return log_prob_c_pos(x, a, c)
    else:
        return log_prob_c_neg(x, a, c)


def compute_target_pdf(x, a, c, normalized=True):
    log_p = log_prob(x, a, c)
    log_p -= np.max(log_p)  # Avoid numerical under-flow when exponentiation.
    prob = np.exp(log_p.astype(float))
    if normalized:
        prob = prob / np.trapz(y=prob, x=x)
    return prob


def orig_target_pdf(x, a, c):
    a_sq = a ** 2.0
    g_x_tmp = - a_sq * (x - c) ** 2.0
    g_x = np.exp(g_x_tmp.astype(float))
    h_x = x / (1.0 + x ** 2.0)
    return h_x * g_x


def compute_orig_target_pdf(x, a, c, normalized=True):
    # log_p = orig_target_pdf(x, a, c)
    # log_p -= np.max(log_p)  # Avoid numerical under-flow when exponentiation.
    # prob = np.exp(log_p)
    prob = orig_target_pdf(x, a, c)
    if normalized:
        prob = prob / np.trapz(y=prob, x=x)
    return prob


def x_to_eta(x):
    return np.sqrt(np.exp(2.0 * x) - 1.0)


def plot_hist_against_target(ax, lscale_samples, a, c):
    # Restrict the plot range; otherwise, the empirical
    # distribution of a heavy-tailed target is unstable.
    max_quantile = .99
    upper_lim = np.quantile(lscale_samples, max_quantile)
    bins = np.linspace(0, upper_lim, 51)

    x = np.linspace(0, upper_lim, 1001)[1:]
    ax.hist(lscale_samples, bins=bins, density=True,
            label='empirical dist')
    ax.plot(x, compute_target_pdf(x, a, c),
            label='target pdf')
    ax.set_xlabel(
        'a= {:1g}, c = {:1g}'.format(a, c))
    ax.set_yticks([])


def plot_hist_against_orig_target(ax, lscale_samples, a, c):
    # Restrict the plot range; otherwise, the empirical
    # distribution of a heavy-tailed target is unstable.
    max_quantile = .99
    eta_samples = x_to_eta(lscale_samples)
    upper_lim = np.quantile(eta_samples, max_quantile)
    bins = np.linspace(0, upper_lim, 51)

    x = np.linspace(0, upper_lim, 1001)[1:]
    ax.hist(eta_samples, bins=bins, density=True,
            label='empirical dist')
    ax.plot(x, compute_orig_target_pdf(x, a, c),
            label='target pdf')
    ax.set_xlabel(
        'a= {:1g}, c = {:1g}'.format(a, c))
    ax.set_yticks([])


def remove_figure_box_edges(ax, sides=None):
    if sides is None:
        sides = ['left', 'right', 'top']
    for side in sides:
        ax.spines[side].set_visible(False)
