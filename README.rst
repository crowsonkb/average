average
=======

Exponentially weighted moving averaging with initialization bias correction.

Usage
-----

.. code:: python

   from average import EWMA

   # Create a scalar running average.
   # beta=0.5 is the smoothing factor.
   avg = EWMA(beta=0.5)
   avg.update(1)
   avg.update(2)
   print(avg.get())

Prints::

    1.6666666666666667

The average is weighted toward the most recent value. That is, its value is ``1 * 1/3 + 2 * 2/3``. The default value for ``beta`` is 0.9, which is reasonable for many uses. Higher smoothing values increase the amount of weight put on older values in the average.

You can also create running averages shaped like NumPy arrays by supplying a shape and dtype (defaults to ``numpy.float64``). For example:

.. code:: python

   avg = EWMA((4, 4), np.float64)
   # or, equivalently:
   avg = EWMA.like(np.eye(4))
   avg.update(np.eye(4))
   print(avg.get())  # Prints a 4x4 identity matrix.

For example, you could use ``EWMA`` to maintain a running average of HxWx3 video frames, if the frames are in NumPy array format or convertible to it.

Notes
-----

The formula for an exponentially weighted moving average with initialization bias correction is from "`Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_" by Kingma and Ba.
