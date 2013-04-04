# -*- coding: utf-8 -*-
import unittest
import os
from obspy.core.util.base import getMatplotlibVersion, NamedTemporaryFile


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

    def test_NamedTemporaryFile_ContextManager(self):
        """
        Tests the automatic closing/deleting of NamedTemporaryFile using the
        context manager.
        """
        content = "burn after writing"
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
            with open(filename) as fh:
                tmp_content = fh.read()
        self.assertEqual(content, tmp_content)
        self.assertFalse(os.path.exists(filename))
        # check that closing/deletion works even when nothing is done with file
        with NamedTemporaryFile() as tf:
            filename = tf.name
        self.assertFalse(os.path.exists(filename))


def suite():
    return unittest.makeSuite(UtilBaseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
