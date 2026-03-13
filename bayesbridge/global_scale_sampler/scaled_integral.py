import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid


class ScaledIntegral:

    def __init__(self, f: callable):
        """Initializing object

        Parameters
        ----------
        f : callable
            The function which will be evaluated
        """
        self.log_func = f
        self.x = np.array([])
        self.raw_log_y = np.array([])
        self.log_scaling_constant = -np.inf
        self.scaled_y = np.array([])
        self.integral_arr = np.array([])

    def check_function_output(self, arr: np.ndarray):
        """Check if a functions output is without nans and infs

        Parameters
        ----------
        arr : numpy array
            The output that will be evaluated for infs and nans
        """
        if any(np.isnan(arr)):
            warnings.warn("Nan's present in function output.")
        if any(np.isinf(arr)):
            warnings.warn("Infinity's or Negative Infinity's in function output.")

    def evaluate_and_rescale(self, new_x: np.ndarray, insert_type):
        """Evaluate y for new x's, adjust scaled y's and integral

        Parameters
        ----------
        new_x : numpy array
            new values of x to be added

        insert_type : string
            How `new_x` should be added. 'append', 'prepend', 'sandwich' or 'interweave.'

        """
        if not all(np.isfinite(new_x)):
            raise ValueError("`new_x` must only contain finite values")
        log_new_y = np.array([self.log_func(x) for x in new_x])
        self.x = self._insert_new(self.x, new_x, insert_type)
        self.raw_log_y = self._insert_new(self.raw_log_y, log_new_y, insert_type)
        max_log_new_y = max(log_new_y)
        if max_log_new_y > self.log_scaling_constant:
            self.log_scaling_constant = max_log_new_y
        self.scaled_y = np.exp(self.raw_log_y - self.log_scaling_constant)
        self.integral_arr = cumulative_trapezoid(y=self.scaled_y, x=self.x, initial=0)

    def _insert_new(self, existing: np.ndarray, new: np.ndarray, insert_type: str):
        """Insert new values from new into existing array

        Parameters
        ----------
        existing : numpy array
            existing values of x

        new : numpy array
            new values of x to be added

        insert_type : string
            How `new` should be added. 'append', 'prepend', 'sandwich' or 'interweave.'
        """
        if insert_type == "interweave":
            if len(new) != len(existing) - 1:
                raise ValueError(
                    "Interweaving lists requires new list be 1 shorter than existing list")
            return np.array([val for pair in zip(existing, new) for val in pair] + [existing[-1]])
        elif insert_type == "append":
            return np.append(existing, new)
        elif insert_type == "sandwich":
            return np.insert(existing, [0, len(existing)], new)
        else:
            raise ValueError("Invalid insert type, must be either \"interweave\" or \"append\"")

    def normalize_integral(self):
        """Normalize scaled_y, log_scaling_constant, and integral_arr so that
        integral_arr[-1] = 1

        """
        self.log_scaling_constant = self.log_scaling_constant + np.log(self.integral_arr[-1])
        self.scaled_y = np.exp(self.raw_log_y - self.log_scaling_constant)
        self.integral_arr = self.integral_arr / self.integral_arr[-1]
