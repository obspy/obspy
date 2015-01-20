# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util.base import getMatplotlibVersion, NamedTemporaryFile
from obspy.core.util.testing import ImageComparison, \
    ImageComparisonException
from obspy.core.util.decorator import skipIf
import os
import unittest
import shutil


# checking for matplotlib
try:
    import matplotlib  # @UnusedImport
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def image_comparison_in_function(path, img_basename, img_to_compare):
    """
    This is just used to wrap an image comparison to check if it raises or not.
    """
    with ImageComparison(path, img_basename, adjust_tolerance=False) as ic:
        shutil.copy(img_to_compare, ic.name)


class UtilBaseTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.base
    """
    @skipIf(not HAS_MATPLOTLIB, 'matplotlib is not installed')
    def test_getMatplotlibVersion(self):
        """
        Tests for the getMatplotlibVersion() function as it continues to cause
        problems.
        """
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

        matplotlib.__version__ = "1.3.1rc2"
        version = getMatplotlibVersion()
        self.assertEqual(version, [1, 3, 1])

        # Set it to the original version str just in case.
        matplotlib.__version__ = original_version

    def test_NamedTemporaryFile_ContextManager(self):
        """
        Tests the automatic closing/deleting of NamedTemporaryFile using the
        context manager.
        """
        content = b"burn after writing"
        # write something to tempfile and check closing/deletion afterwards
        with NamedTemporaryFile() as tf:
            filename = tf.name
            tf.write(content)
        self.assertFalse(os.path.exists(filename))
        # write something to tempfile and check that it is written correctly
        with NamedTemporaryFile() as tf:
            filename = tf.name
            tf.write(content)
            tf.close()
            with open(filename, 'rb') as fh:
                tmp_content = fh.read()
        self.assertEqual(content, tmp_content)
        self.assertFalse(os.path.exists(filename))
        # check that closing/deletion works even when nothing is done with file
        with NamedTemporaryFile() as tf:
            filename = tf.name
        self.assertFalse(os.path.exists(filename))

    def test_image_comparison(self):
        """
        Tests the image comparison mechanism with an expected fail and an
        expected passing test.
        Also tests that temporary files are deleted after both passing and
        failing tests.
        """
        path = os.path.join(os.path.dirname(__file__), "images")
        img_basename = "image.png"
        img_ok = os.path.join(path, "image_ok.png")
        img_fail = os.path.join(path, "image_fail.png")

        # image comparison that should pass
        with ImageComparison(path, img_basename) as ic:
            shutil.copy(img_ok, ic.name)
            self.assertTrue(os.path.exists(ic.name))
        # check that temp file is deleted
        self.assertFalse(os.path.exists(ic.name))

        # image comparison that should raise
        self.assertRaises(ImageComparisonException,
                          image_comparison_in_function, path, img_basename,
                          img_fail)
        # check that temp file is deleted
        self.assertFalse(os.path.exists(ic.name))


def suite():
    return unittest.makeSuite(UtilBaseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
