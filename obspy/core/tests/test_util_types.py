# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util import Enum, ComplexWithUncertainties, \
    FloatWithUncertainties
import unittest


class UtilTypesTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.base
    """
    def test_enum(self):
        """
        Tests for the enum type.
        """
        items = ["m", "s", "m/s", "m/(s*s)", "m*s", "dimensionless", "other"]
        units = Enum(items)
        # existing selections
        self.assertEqual(units.other, "other")
        self.assertEqual(units.M, "m")
        self.assertEqual(units['m/s'], "m/s")
        self.assertEqual(units.get('m/s'), "m/s")
        self.assertEqual(units[0], "m")
        self.assertEqual(units[-1], "other")
        # not existing selections should fail
        self.assertRaises(Exception, units.__getitem__, '5')
        self.assertRaises(Exception, units.__getattr__, 'xx')
        self.assertRaises(Exception, units.get, 'xx', 'default')
        self.assertRaises(Exception, units.__getitem__, 99)
        self.assertRaises(Exception, units.__getitem__, -99)
        # test in operator
        self.assertTrue("other" in units)
        self.assertTrue("ot21her" not in units)
        # test typical dict methods
        self.assertEqual(units.values(), items)
        self.assertEqual(units.items(), list(zip(items, items)))
        self.assertEqual(units.keys(), items)
        # call will either return correct enum label or return None
        self.assertEqual(units('m'), 'm')
        self.assertEqual(units('m/(s*s)'), 'm/(s*s)')
        self.assertEqual(units(5), 'dimensionless')
        self.assertEqual(units(99), None)
        self.assertEqual(units('xxx'), None)

    def _check_complex_with_u(self, c, real, r_lo, r_up, imag, i_lo, i_up):
        """
        Check for six equalities for a ComplexWithUncertainties
        """
        self.assertTrue(isinstance(c.real, FloatWithUncertainties))
        self.assertTrue(isinstance(c.imag, FloatWithUncertainties))
        self.assertEqual(c.real, real)
        self.assertEqual(c.imag, imag)
        self.assertEqual(c.real.upper_uncertainty, r_up)
        self.assertEqual(c.real.lower_uncertainty, r_lo)
        self.assertEqual(c.imag.upper_uncertainty, i_up)
        self.assertEqual(c.imag.lower_uncertainty, i_lo)

    def test_complex(self):
        """
        Test the ComplexWithUncertainties for proper usage
        """
        f1 = float(3.5)
        f2 = float(12)
        lu1 = 1
        uu1 = 2
        lu2 = 4.1
        uu2 = 7.2
        fu1 = FloatWithUncertainties(f1, lower_uncertainty=lu1,
                                     upper_uncertainty=uu1)
        fu2 = FloatWithUncertainties(f2, lower_uncertainty=lu2,
                                     upper_uncertainty=uu2)
        c1 = ComplexWithUncertainties()
        c2 = ComplexWithUncertainties(f1, f2)
        c3 = ComplexWithUncertainties(
            f1, f2, lower_uncertainty=complex(lu1, lu2),
            upper_uncertainty=complex(uu1, uu2))
        c4 = ComplexWithUncertainties(fu1, fu2)
        # c1 should be 0+0j with uncertainties of None
        self._check_complex_with_u(c1, 0, None, None, 0, None, None)
        # c2 should return the floats
        self._check_complex_with_u(c2, f1, None, None, f2, None, None)
        # c3 and c4 should be the same.
        self._check_complex_with_u(c3, f1, lu1, uu1, f2, lu2, uu2)
        self._check_complex_with_u(c4, f1, lu1, uu1, f2, lu2, uu2)
        self.assertEqual(c4.real, fu1)
        self.assertEqual(c4.imag, fu2)


def suite():
    return unittest.makeSuite(UtilTypesTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
