#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The paz test suite.
"""
import io
import unittest

from obspy.io.gse2 import paz


class PAZTestCase(unittest.TestCase):
    """
    Test cases for reading GSE PAZ files.
    """
    def test_read_with_spaces(self):
        """
        Reading PAZ files where PAZ are separated by spaces.
        """
        f = io.StringIO("""CAL1 RJOB   LE-3D    Z  M24    PAZ 010824 0001
        2
        -4.39823 4.48709
        -4.39823 -4.48709
        3
        0.0 0.0
        0.0 0.0
        0.0 0.0
        0.4""")
        p, z, k = paz.read_paz(f)
        self.assertAlmostEqual(-4.39823, p[0].real)
        self.assertAlmostEqual(4.48709, p[0].imag)
        self.assertAlmostEqual(-4.39823, p[1].real)
        self.assertAlmostEqual(-4.48709, p[1].imag)
        self.assertEqual([0j, 0j, 0j], z)
        self.assertAlmostEqual(0.4, k)
        f.close()

    def test_read_without_spaces(self):
        """
        Reading PAZ files where PAZ are not separated by spaces.

        Tests uses unrealistic PAZ information.
        """
        f = io.StringIO("""CAL1 RJOB   LE-3D    Z  M24    PAZ 010824 0001
2
-4.3982340.48709
-4.39823-4.48709
3
1.2 4.0
-1.09823-3.08709
-1.0982330.08709
0.5""")
        p, z, k = paz.read_paz(f)
        self.assertAlmostEqual(-4.39823, p[0].real)
        self.assertAlmostEqual(40.48709, p[0].imag)
        self.assertAlmostEqual(-4.39823, p[1].real)
        self.assertAlmostEqual(-4.48709, p[1].imag)
        self.assertAlmostEqual(1.2, z[0].real)
        self.assertAlmostEqual(4.0, z[0].imag)
        self.assertAlmostEqual(-1.09823, z[1].real)
        self.assertAlmostEqual(-3.08709, z[1].imag)
        self.assertAlmostEqual(-1.09823, z[2].real)
        self.assertAlmostEqual(30.08709, z[2].imag)
        self.assertAlmostEqual(0.5, k)
        f.close()
