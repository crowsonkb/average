average
=======

Moving averaging schemes (exponentially weighted, polynomial-decay). Running averages only are maintained; no history of values is stored.

Usage (EWMA; exponentially weighted moving average)
---------------------------------------------------

.. code:: python

   from average import EWMA

   # Create a scalar running average.
   # beta=0.5 is the smoothing factor.
   avg = EWMA(beta=0.5)
   avg.update(1)
   avg.update(2)
   print(avg.get())  # Prints 1.6666666666666667.

The average is weighted toward the most recent value. That is, its value is ``1 * 1/3 + 2 * 2/3``. The default value for ``beta`` is 0.9, which is reasonable for many uses. Higher smoothing values increase the amount of weight put on less recent values in the average.

You can also create running averages shaped like NumPy arrays by supplying a shape and dtype (defaults to ``numpy.float64``). For example:

.. code:: python

   avg = EWMA((4, 4), np.float64)
   # or, equivalently:
   avg = EWMA.like(np.eye(4))
   avg.update(np.eye(4))
   print(avg.get())  # Prints a 4x4 identity matrix.

For example, you could use ``EWMA`` to maintain a running average of HxWx3 video frames, if the frames are in NumPy array format or convertible to it.

Usage (PDMA; polynomial-decay moving average)
---------------------------------------------

With the polynomial decay parameter ``eta`` set to the default value of 0, ``PDMA`` acts as a simple average (averages equally over all previous values). A history of values is not kept.

.. code:: python

   from average import PDMA

   avg = PDMA()
   avg.update(1)
   avg.update(2)
   avg.update(3)
   print(avg.get())  # Prints 2.0.

Higher values of ``eta`` correspond to a polynomially decaying window of degree ``eta`` stretching back over all previous values. The higher ``eta`` is set, the more weight is placed on more recent values.

.. code:: python

   avg = PDMA(eta=1)
   for i in range(1, 5):
      avg.update(i)
   print(avg.get())  # Prints 3.0.

In our example, setting ``eta`` to 0 would instead have printed the simple average 2.5. ``eta`` can be set arbitrarily high, but 0, 1, and 3 are probably reasonable values for many uses. Similarly to ``EWMA``, ``PDMA`` running averages can be shaped like NumPy arrays and have a NumPy data type (not shown).

References
----------

The formula for an exponentially weighted average with initialization bias correction is given in "`Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_" by Kingma and Ba.

The formula for polynomial-decay averaging is given in section 4 of "`Stochastic Gradient Descent for Non-smooth Optimization: Convergence Results and Optimal Averaging Schemes <http://proceedings.mlr.press/v28/shamir13.pdf>`_" by Shamir and Zhang.
