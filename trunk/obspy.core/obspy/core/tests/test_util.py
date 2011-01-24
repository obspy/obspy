# -*- coding: utf-8 -*-

from obspy.core import path
import unittest
import os

class UtilTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util
    """

    def test_path(self):
        """
        Tests if path returns path to testfile
        """
        file = path('slist.ascii')
        self.assertEquals(['obspy', 'core', 'tests', 'data',
                           'slist.ascii'], file.split(os.path.sep)[-5:])
        # check that exception is raised on nonexisiting filename
        self.assertRaises(IOError, path, 'this_file_does_not_exist.ascii')


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
