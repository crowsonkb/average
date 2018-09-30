"""Unit tests for average.EWMA."""

import unittest

import numpy as np

from average import EWMA


class Test(unittest.TestCase):
    def test_scalar_once(self):
        avg = EWMA()
        avg.update(1)
        self.assertEqual(avg.get(), 1)

    def test_scalar_twice(self):
        avg = EWMA(beta=0.5)
        avg.update(0)
        avg.update(1)
        self.assertEqual(avg.get(), 2/3)

    def test_array_once(self):
        avg = EWMA((4, 4))
        avg.update(np.eye(4))
        self.assertTrue((avg.get() == np.eye(4)).all())

    def test_returns_float(self):
        avg = EWMA()
        avg.update(1)
        self.assertEqual(type(avg.get()), float)

    def test_raises_on_wrong_shape(self):
        avg = EWMA((4, 4))
        with self.assertRaises(ValueError):
            avg.update(np.eye(3))

    def test_raises_on_wrong_type(self):
        avg = EWMA((4, 4))
        with self.assertRaises(TypeError):
            avg.update('abc')

    def test_returns_scalar_nan(self):
        avg = EWMA()
        out = avg.get()
        self.assertNotEqual(out, out)

    def test_returns_array_nan(self):
        avg = EWMA((4, 4))
        out = avg.get()
        self.assertTrue(np.isnan(out).all())

    def test_get_est(self):
        avg = EWMA(beta=0.5)
        avg.update(0)
        out = avg.get_est(1)
        self.assertEqual(out, 2/3)

    def test_like(self):
        arr = np.float32(np.eye(4))
        avg = EWMA.like(arr)
        out = avg.update(np.eye(4))
        self.assertEqual(out.shape, arr.shape)
        self.assertEqual(out.dtype, arr.dtype)


if __name__ == '__main__':
    unittest.main()
