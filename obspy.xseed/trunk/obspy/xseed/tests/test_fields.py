# -*- coding: utf-8 -*-

import unittest

from obspy.xseed.fields import Float


class FieldsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_formatExponential(self):
        field = Float(1, "test", 12, mask='%+1.5e')
        self.assertEquals(field.write('2.5'), '+2.50000E+00')


def suite():
    return unittest.makeSuite(FieldsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
