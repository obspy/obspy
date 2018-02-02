# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.core.compatibility import py3_round


class CompatibilityTestCase(unittest.TestCase):
    def test_py3_round(self):
        """
        Ensure py3_round rounds correctly and returns expected data types.
        """
        # list of tuples; (input, ndigits, expected, excpected type)
        test_list = [
            (.222, 2, .22, float),
            (1516047903968282880, -3, 1516047903968283000, int),
            (1.499999999, None, 1, int),
            (.0222, None, 0, int),
            (12, -1, 10, int),
            (15, -1, 20, int),
            (15, -2, 0, int),
        ]
        for number, ndigits, expected, expected_type in test_list:
            # this is a fix for py3.4 which cannot take None as ndigits
            if ndigits is not None:
                out = py3_round(number, ndigits)
            else:
                out = py3_round(number)
            self.assertEqual(out, expected)
            self.assertIsInstance(out, expected_type)


def suite():
    return unittest.makeSuite(CompatibilityTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
