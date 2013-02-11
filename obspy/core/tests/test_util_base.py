# -*- coding: utf-8 -*-
from obspy.core.util.base import getMatplotlibVersion
import unittest


class UtilBaseTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.base
    """
    def test_getMatplotlibVersion(self):
        """
        Tests for the getMatplotlibVersion() function as it continues to cause
        problems.
        """
        import matplotlib
        original_version = matplotlib.__version__

        matplotlib.__version__ = "1.2.3"
        version = getMatplotlibVersion()
        self.assertEqual(version, [1, 2, 3])
        matplotlib.__version__ = "0.9.11"
        version = getMatplotlibVersion()
        self.assertEqual(version, [0, 9, 11])

        matplotlib.__version__ = "0.9.svn"
        version = getMatplotlibVersion()
        self.assertEqual(version, [0, 9, 0])

        matplotlib.__version__ = "1.1.1~rc1-1"
        version = getMatplotlibVersion()
        self.assertEqual(version, [1, 1, 1])

        matplotlib.__version__ = "1.2.x"
        version = getMatplotlibVersion()
        self.assertEqual(version, [1, 2, 0])

        # Set it to the original version str just in case.
        matplotlib.__version__ = original_version


def suite():
    return unittest.makeSuite(UtilBaseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
