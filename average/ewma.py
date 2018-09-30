"""Exponentially weighted moving averaging with initialization bias correction."""

from numbers import Number

import numpy as np


class EWMA:
    """Exponentially weighted moving averaging with initialization bias correction.

    :param shape: The NumPy array shape to use. Defaults to an empty tuple (a scalar).
    :type shape: tuple[int]
    :param dtype: The NumPy data type to use. Defaults to float64.
    :type dtype: numpy.dtype
    :param beta: The smoothing factor to use. Must be between 0 and 1.
    :type beta: float
    :param correct_bias: Whether to correct the running average's initialization bias.
    :type correct_bias: bool
    """
    def __init__(self, shape=(), dtype=float, beta=0.9, correct_bias=True):
        self.beta = beta
        self.beta_accum = 1 if correct_bias else 0
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

        :returns: The current value of the running average. If update() has never been called,
            returns NaN.
        :rtype: float or numpy.ndarray
        """
        with np.errstate(invalid='ignore'):
            out = self.value / (1 - self.beta_accum)
        return self._de_numpy(out)

    def get_est(self, datum):
        """Estimates the next value of the running average given a potential new value. Does not
        update the average.

        :param datum: The potential new value.
        :type datum: float or numpy.ndarray
        :returns: The predicted next value of the running average given the potential new value.
        :rtype: float or numpy.ndarray
        """
        self._verify_input(datum)
        est_value = self.beta * self.value + (1 - self.beta) * datum
        out = est_value / (1 - self.beta_accum * self.beta)
        return self._de_numpy(out)

    @classmethod
    def like(cls, arr, beta=0.9, correct_bias=True):
        """Creates a new EWMA object with its shape and dtype like the specified NumPy array.

        :param arr: The array to take the shape and data type from.
        :type arr: numpy.ndarray
        :param beta: The smoothing factor to use. Must be between 0 and 1.
        :type beta: float
        :param correct_bias: Whether to correct the running average's initialization bias.
        :type correct_bias: bool
        :returns: The new exponentially weighted moving average object.
        :rtype: average.EWMA
        """
        return cls(arr.shape, arr.dtype, beta, correct_bias)

    def update(self, datum):
        """Updates the running average with a new observation.

        :param datum: The new value.
        :type datum: float or numpy.ndarray
        :returns: The value of the running average after updating it with the new value.
        :rtype: float or numpy.ndarray
        """
        self._verify_input(datum)
        self.beta_accum *= self.beta
        self.value *= self.beta
        self.value += (1 - self.beta) * datum
        return self.get()
