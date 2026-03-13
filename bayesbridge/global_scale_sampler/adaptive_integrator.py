import warnings
import numpy as np

from .scaled_integral import ScaledIntegral


class AdaptiveIntegrator:
    """Class for refining integration of ScaledIntegral objects"""

    def __init__(self, f: ScaledIntegral, starting_point: float, bound_threshold: float, step_size_threshold: float,
                 original_step_size: float, max_bound_iters: int, max_step_iters: int):
        """Initialize AdaptiveIntegrator object"""
        self.f = f
        self.bound_threshold = bound_threshold
        self.step_size_threshold = step_size_threshold
        self.original_step_size = original_step_size
        self.starting_point = starting_point
        self.max_bound_iters = max_bound_iters
        self.max_step_iters = max_step_iters

    def integrate(self):
        bound_refined = refine_bounds(self.f, self.original_step_size, self.starting_point,
                                      self.bound_threshold, self.max_bound_iters)
        return refine_step_size(bound_refined, self.step_size_threshold, self.max_step_iters)


def abs_percent_change(old, new):
    if np.isclose(old, 0):
        return np.inf
    return abs(new - old) / old


def refine_step_size(f, threshold, max_iters):
    """Refine step size

        Parameters
        ----------
        f: ScaledIntegral object with multiple x values so far
        threshold: float that denotes how much change is enough for more iterations
        max_iters: stop integrating after max_iters

        Returns
        --------
        Refined f
    """
    for _ in range(max_iters):
        old_integral, old_scaling_constant = f.integral_arr[-1], f.log_scaling_constant
        midpoints = (f.x[1:] + f.x[:-1]) / 2
        f.evaluate_and_rescale(midpoints, "interweave")
        new_integral = f.integral_arr[-1]
        rescaled_old_integral = old_integral * np.exp(old_scaling_constant - f.log_scaling_constant)
        if abs_percent_change(rescaled_old_integral, new_integral) < threshold:
            return f
    warnings.warn(f"Max iterations {max_iters} reached in step size refinement")
    return f


def refine_bounds(f, step_size, starting_point, threshold, max_iters, tail_threshold=1e-4):
    """Refine the bounds of ScaledIntegral object until difference in
        integrals is less than bound_threshold

        Parameters
        ----------
        f: ScaledIntegral object to refine
        step_size: Existing step size
        starting_point: Where to begin refining
        threshold: float denoting the decimal change necessary to continue refining
        max_iters: Maximum number of iterations
        tail_threshold: float determining maximum value on either tail end before integration can stop

        Returns
        ------
        Updated ScaledIntegral object
    """
    f.evaluate_and_rescale(np.array([starting_point]), "append")
    for i in range(1, max_iters + 1):
        old_integral, old_scaling_constant = f.integral_arr[-1], f.log_scaling_constant
        f.evaluate_and_rescale([starting_point - i * step_size, starting_point + i * step_size],
                               "sandwich")
        new_integral = f.integral_arr[-1]
        rescaled_old_integral = old_integral * np.exp(old_scaling_constant - f.log_scaling_constant)
        if (
                i != 1 and
                abs_percent_change(rescaled_old_integral, new_integral) < threshold and
                f.scaled_y[-1] < tail_threshold and
                f.scaled_y[0] < tail_threshold
        ):
            return f
    warnings.warn(f"Max iterations {max_iters} reached in bound refinement")
    return f
