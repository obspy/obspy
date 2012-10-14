# -*- coding: utf-8 -*-

from obspy.core.util.types import Enum
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
        self.assertEquals(units.other, "other")
        self.assertEquals(units.M, "m")
        self.assertEquals(units['m/s'], "m/s")
        self.assertEquals(units.get('m/s'), "m/s")
        self.assertEquals(units[0], "m")
        self.assertEquals(units[-1], "other")
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
        self.assertEquals(units.values(), items)
        self.assertEquals(units.items(), zip(items, items))
        self.assertEquals(units.keys(), items)
        # call will either return correct enum label or return None
        self.assertEquals(units('m'), 'm')
        self.assertEquals(units('m/(s*s)'), 'm/(s*s)')
        self.assertEquals(units(5), 'dimensionless')
        self.assertEquals(units(99), None)
        self.assertEquals(units('xxx'), None)


def suite():
    return unittest.makeSuite(UtilTypesTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
