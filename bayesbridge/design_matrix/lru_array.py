import numpy as np
from functools import lru_cache, wraps

def np_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # convert ndarray args → (tuple, shape)
        def convert(a):
            if isinstance(a, np.ndarray):
                return (tuple(a.ravel()), a.shape)
            return a

        new_args = tuple(convert(a) for a in args)
        new_kwargs = {k: convert(v) for k, v in kwargs.items()}
        return _cached_func(*new_args, **new_kwargs)

    @lru_cache(maxsize=16)
    def _cached_func(*args, **kwargs):
        # convert (tuple, shape) back into array
        def recover(a):
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], tuple):
                return np.array(a[0], dtype=float).reshape(a[1])
            return a

        rec_args = tuple(recover(a) for a in args)
        rec_kwargs = {k: recover(v) for k, v in kwargs.items()}
        return func(*rec_args, **rec_kwargs)

    return wrapper