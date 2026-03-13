from warnings import warn

import numpy as np
from .abstract_matrix import AbstractDesignMatrix
from .lru_array import np_cache


class DenseDesignMatrix(AbstractDesignMatrix):
    
    def __init__(self, X, center_predictor=False, add_intercept=True,
                 copy_array=False):
        """
        Params:
        ------
        X : numpy array
        """
        self.use_cupy = False
        if copy_array:
            X = X.copy()
        super().__init__()
        X = self.remove_intercept_indicator(X)
        if center_predictor:
            X -= np.mean(X, axis=0)[np.newaxis, :]
        self.X_main = np.copy(X)
        if add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.intercept_added = add_intercept
        self.centered = center_predictor
        self.unweighted_fisher_info = None
        self.unweighted_fisher_diag = None

    @property
    def shape(self):
        return self.X.shape

    @property
    def is_sparse(self):
        return False

    def dot(self, v):

        if self.memoized and np.all(self.v_prev == v):
            return self.X_dot_v

        result = self.X.dot(v)
        if self.memoized:
            self.X_dot_v = result
            self.v_prev = v
        self.dot_count += 1

        return result

    def Tdot(self, v):
        self.Tdot_count += 1
        return self.X.T.dot(v)

    def main_Tdot(self, v):
        return self.X_main.T.dot(v)

    @np_cache
    def compute_fisher_info(self, weight, diag_only=False):
        """
        Compute $X^T W X$ where W is the diagonal matrix of a given weight.
        If diag_only == True, it returns only the diagonal of the output matrix.
        """
        if np.isscalar(weight):
            return weight * self.compute_unweighted_fisher_info(diag_only)
        if diag_only:
            return np.sum(weight[:, np.newaxis] * self.X ** 2, 0)
        else:
            return self.X.T.dot(weight[:, np.newaxis] * self.X)

    def compute_unweighted_fisher_info(self, diag_only):
        if diag_only:
            if self.unweighted_fisher_diag is None:
                self.unweighted_fisher_diag = np.sum(self.X ** 2, 0)
            return self.unweighted_fisher_diag
        else:
            if self.unweighted_fisher_info is None:
                # Cache the result when computing for the first time.
                self.unweighted_fisher_info = self.X.T.dot(self.X)
            return self.unweighted_fisher_info

    @np_cache
    def compute_transposed_fisher_info(self, weight, include_intrcpt=False):
        # TODO: Check implementation.
        # Note: with current implementation of the class, `self.X` explicitly includes
        # the intercept when `self.intercept_added == True`.
        """
        Compute $X W X^T$ where W is the diagonal matrix of a given weight.
        """
        if not include_intrcpt and self.intercept_added:
            weight = np.concatenate(([0], weight))
        return self.X.dot(weight[:, np.newaxis] * self.X.T)

    def toarray(self):
        return self.X

    def extract_matrix(self, order=None):
        return self.X
