#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The paz test suite.
"""

from obspy.gse2 import paz
import numpy as np
import os
import unittest
import StringIO


class PAZTestCase(unittest.TestCase):
    """
    Test cases for reading GSE PAZ files.
    """
    def test_readWithSpace(self):
        """
        Reading PAZ files where PAZ are separated by spaces.
        """
        f = StringIO.StringIO("""CAL1 RJOB   LE-3D    Z  M24    PAZ 010824 0001
        2
        -4.39823 4.48709
        -4.39823 -4.48709
        3
        0.0 0.0
        0.0 0.0
        0.0 0.0
        0.4""")
        p, z, k = paz.readPaz(f)
        self.assertAlmostEquals(-4.39823 + 4.48709j, p[0])
        self.assertAlmostEquals(-4.39823 - 4.48709j, p[1])
        self.assertEquals([0j, 0j, 0j], z)
        self.assertAlmostEquals(0.4, k)
        f.close()

    def test_readWithOutSpace(self):
        """
        Reading PAZ files where PAZ are not separated by spaces.

        Tests uses unrealistic PAZ information.
        """
        f = StringIO.StringIO("""CAL1 RJOB   LE-3D    Z  M24    PAZ 010824 0001
2
-4.3982340.48709
-4.39823-4.48709
3
1.2 4.0
-1.09823-3.08709
-1.0982330.08709
0.5""")
        p, z, k = paz.readPaz(f)
        self.assertAlmostEquals(-4.39823 + 40.48709j, p[0])
        self.assertAlmostEquals(-4.39823 - 4.48709j, p[1])
        self.assertAlmostEquals(1.2 + 4.0j, z[0])
        self.assertAlmostEquals(-1.09823 - 3.08709j, z[1])
        self.assertAlmostEquals(-1.09823 + 30.08709j, z[2])
        self.assertAlmostEquals(0.5, k)
        f.close()


def suite():
    return unittest.makeSuite(PAZTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
