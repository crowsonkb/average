"""Unit tests for average.PDMA."""

import unittest

import numpy as np

from average import PDMA


class Test(unittest.TestCase):
    def test_scalar_once(self):
        avg = PDMA()
        avg.update(1)
        self.assertEqual(avg.get(), 1)

    def test_scalar_twice(self):
        avg = PDMA()
        avg.update(0)
        avg.update(1)
        self.assertEqual(avg.get(), 0.5)

    def test_scalar_twice_nonzero_eta(self):
        avg = PDMA(eta=1)
        avg.update(0)
        avg.update(1)
        self.assertEqual(avg.get(), 2/3)

    def test_array_once(self):
        avg = PDMA((4, 4))
        avg.update(np.eye(4))
        self.assertTrue((avg.get() == np.eye(4)).all())

    def test_returns_float(self):
        avg = PDMA()
        avg.update(1)
        self.assertEqual(type(avg.get()), float)

    def test_raises_on_wrong_shape(self):
        avg = PDMA((4, 4))
        with self.assertRaises(ValueError):
            avg.update(np.eye(3))

    def test_raises_on_wrong_type(self):
        avg = PDMA((4, 4))
        with self.assertRaises(TypeError):
            avg.update('abc')

    def test_returns_scalar_zero(self):
        avg = PDMA()
        out = avg.get()
        self.assertEqual(out, 0)

    def test_returns_array_zero(self):
        avg = PDMA((4, 4))
        out = avg.get()
        self.assertTrue((out == 0).all())

    def test_like(self):
        arr = np.float32(np.eye(4))
        avg = PDMA.like(arr)
        out = avg.update(np.eye(4))
        self.assertEqual(out.shape, arr.shape)
        self.assertEqual(out.dtype, arr.dtype)


if __name__ == '__main__':
    unittest.main()
