# -*- coding: utf-8 -*-

from obspy.core.util import Enum
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
        self.assertEqual(units.items(), zip(items, items))
        self.assertEqual(units.keys(), items)
        # call will either return correct enum label or return None
        self.assertEqual(units('m'), 'm')
        self.assertEqual(units('m/(s*s)'), 'm/(s*s)')
        self.assertEqual(units(5), 'dimensionless')
        self.assertEqual(units(99), None)
        self.assertEqual(units('xxx'), None)


def suite():
    return unittest.makeSuite(UtilTypesTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
