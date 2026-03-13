import bisect
import warnings

import numpy as np

from .adaptive_integrator import AdaptiveIntegrator
from .scaled_integral import ScaledIntegral


class InverseCDFSampler():

    def __init__(self, f, starting_point, rg, bound_threshold=1e-3, step_size_threshold=1e-3,
                 original_step_size=1, max_bound_iters=20, max_step_iters=10):
        """Initialize for a function evaluated on log scale."""
        self.rg = rg
        self.si = ScaledIntegral(f)
        ad_int = AdaptiveIntegrator(self.si, starting_point, bound_threshold, step_size_threshold, original_step_size,
                                    max_bound_iters, max_step_iters)
        self.si = ad_int.integrate()
        self.si.normalize_integral()

    def bounded_quadratic_solver(self, a, b, c, lower_bound, upper_bound):
        """
        Solve ax^2 + bx + c = 0 for a value x between lower_bound and upper_bound

        Parameters
        ----------
        a: Leading coefficient
        b: Linear coefficient
        c: Scalar coefficient
        lower_bound: Only return roots higher than lower_bound
        upper_bound: Only return roots lower than upper_bound
        Returns
        -------
        A single root
        """
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            warnings.warn(f"{a}x^2 + {b}x + {c} = 0 has no real roots")
            return np.nan

        elif discriminant == 0:
            return -1 * b / (2 * a)

        else:
            if np.isclose(a, 0):
                return -c / b
            else:
                root1 = (-1 * b + np.sqrt(discriminant)) / (2 * a)
                root2 = (-1 * b - np.sqrt(discriminant)) / (2 * a)
            upper_root = max(root1, root2)
            lower_root = min(root1, root2)

        if upper_root > upper_bound and lower_root < lower_bound:
            warnings.warn(
                f"{a}x^2 + {b}x + {c} = 0 has no roots between {lower_bound} and {upper_bound}")
            return np.nan

        elif upper_root > upper_bound:
            return lower_root

        elif lower_root < lower_bound:
            return upper_root

        else:
            warnings.warn(
                f"{a}x^2 + {b}x + {c} = 0 has two roots between {lower_bound} and {upper_bound}. "
                f"Upper value returned")
            return upper_root

    def find_between(self, u, ind1, ind2):
        """
        Find F-inv(u) where x[ind1] and x[ind2] are adjacent values in the ScaledIntegral's x array
        for which F(x1) < u and F(x2) > u

        Parameters
        ---------
        u: The value for which we want F-inv(u)
        ind1: the largest index for which integral_arr[index] < u
        ind2: the smallest index for which integral_arr[index] > u

        Returns
        -------
        x for which F(x) is close to u
        """
        F1 = self.si.integral_arr[ind1]
        f1 = self.si.scaled_y[ind1]
        f2 = self.si.scaled_y[ind2]
        x1 = self.si.x[ind1]
        x2 = self.si.x[ind2]
        diff = u - F1
        slope = (f2 - f1) / (x2 - x1)
        a = slope / 2
        b = f1
        c = -1 * diff
        return self.bounded_quadratic_solver(a, b, c, 0, x2 - x1) + x1

    def sample(self):
        """Sample once from the Inverse CDF sampler."""
        u = self.rg.np_random.uniform(0, 1)
        return self.F_inv(u)

    def F_inv(self, u):
        """Return F_inv(u)

        Parameters
        ----------
        u: The value for which we return F-inv(u).

        Returns
        -------
        F-inv(u)
        """
        idx = bisect.bisect(self.si.integral_arr, u)
        return self.find_between(u, idx - 1, idx) if idx < len(self.si.x) else self.si.x[-1]
