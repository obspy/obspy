# -*- coding: utf-8 -*-
"""
The seisan.core test suite.
"""

from obspy.seisan.core import _getVersion, isSEISAN, readSEISAN
import inspect
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    Test cases for SEISAN core interfaces.
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(self.dir, 'data')

    def tearDown(self):
        pass

    def test_getVersion(self):
        """
        Tests resulting version strings of SEISAN file.
        """
        # 1
        file = os.path.join(self.path, '1996-06-03-1917-52S.TEST__002')
        data = open(file, 'rb').read(80 * 12)
        self.assertEqual(_getVersion(data), ('SUN', 32, 7))
        # 2
        file = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        data = open(file, 'rb').read(80 * 12)
        self.assertEqual(_getVersion(data), ('PC', 32, 7))

    def test_isSEISAN(self):
        """
        Tests SEISAN file check.
        """
        # 1
        file = os.path.join(self.path, '1996-06-03-1917-52S.TEST__002')
        self.assertTrue(isSEISAN(file))
        # 2
        file = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        self.assertTrue(isSEISAN(file))

    def test_readSEISAN(self):
        """
        Test SEISAN file reader.
        """
        # 1
        file = os.path.join(self.path, '9701-30-1048-54S.MVO_21_1')
        stream = readSEISAN(file)
        # 2
        #file = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        #self.assertTrue(isSEISAN(file))



def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
