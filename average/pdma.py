"""Polynomial-decay moving averaging."""

from numbers import Number

import numpy as np


class PDMA:
    """Polynomial-decay moving averaging.

    :param shape: The NumPy array shape to use. Defaults to an empty tuple (a scalar).
    :type shape: tuple[int]
    :param dtype: The NumPy data type to use. Defaults to float64.
    :type dtype: numpy.dtype
    :param eta: The polynomial decay factor. Must be >= 0. The default of 0 corresponds to the
        simple average. Higher values of eta correspond to a polynomially decaying window of
        degree eta.
    :type eta: float
    """
    def __init__(self, shape=(), dtype=float, eta=0):
        self.eta = eta
        self.t = 0
        self.value = np.zeros(shape, np.dtype(dtype))

    @staticmethod
    def _de_numpy(arr):
        """Converts NumPy scalars of type float64 to the Python float type."""
        if isinstance(arr, (np.generic, np.ndarray)):
            if not arr.shape and arr.dtype == np.dtype(np.float64):
                return float(arr)
        return arr

    def _verify_input(self, arr):
        """Sanity-checks the input value."""
        if isinstance(arr, Number):
            arr = np.array(arr)
        if not isinstance(arr, (np.generic, np.ndarray)):
            raise TypeError('Input must be Python number or NumPy array')
        if arr.shape != self.value.shape:
            raise ValueError('Shape of input {} must match shape of running average {}'.format(
                arr.shape, self.value.shape
            ))

    def get(self):
        """Gets the current value of the running average.

        :returns: The current value of the running average.
        :rtype: float or numpy.ndarray
        """
        return self._de_numpy(self.value)

    @classmethod
    def like(cls, arr, eta=0):
        """Creates a new PDMA object with its shape and dtype like the specified NumPy array.

        :param arr: The array to take the shape and data type from.
        :type arr: numpy.ndarray
        :param eta: The polynomial decay factor. Must be >= 0. The default of 0 corresponds to the
            simple average. Higher values of eta correspond to a polynomially decaying window of
            degree eta.
        :type eta: float
        :returns: The new polynomial-decay moving average object.
        :rtype: average.PDMA
        """
        return cls(arr.shape, arr.dtype, eta)

    def update(self, datum):
        """Updates the running average with a new observation.

        :param datum: The new value.
        :type datum: float or numpy.ndarray
        :returns: The value of the running average after updating it with the new value.
        :rtype: float or numpy.ndarray
        """
        self._verify_input(datum)
        self.t += 1
        if self.eta >= 0:
            weight = (1 + self.eta) / (self.t + self.eta)
            self.value *= 1 - weight
            self.value += weight * datum
        else:
            self.value[...] = datum
        return self.get()
