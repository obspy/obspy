import os
import unittest

import numpy as np

from obspy import Stream, UTCDateTime
from obspy.io.pdas import _is_pdas, _read_pdas


class PDASTestCase(unittest.TestCase):
    """
    Test suite for pdas.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.testfile = os.path.join(self.path, 'data', 'p1246001.108')

    def test_read_pdas(self):
        """
        Tests the _read_pdas function.
        """
        st = _read_pdas(self.testfile)
        self.assertTrue(isinstance(st, Stream))
        self.assertEqual(len(st), 1)
        tr = st[0]
        expected = [('COMMENT', 'GAINRANGED'),
                    ('DATASET', 'P1246001108'),
                    ('FILE_TYPE', 'LONG'),
                    ('HORZ_UNITS', 'Sec'),
                    ('SIGNAL', 'Channel1'),
                    ('VERSION', 'next'),
                    ('VERT_UNITS', 'Counts')]
        self.assertEqual(sorted(tr.stats.pop("pdas").items()), expected)
        expected = [('_format', 'PDAS'),
                    ('calib', 1.0),
                    ('channel', ''),
                    ('delta', 0.005),
                    ('endtime', UTCDateTime(1994, 4, 18, 0, 0, 2, 495000)),
                    ('location', ''),
                    ('network', ''),
                    ('npts', 500),
                    ('sampling_rate', 200.0),
                    ('starttime', UTCDateTime(1994, 4, 18, 0, 0)),
                    ('station', '')]
        self.assertEqual(sorted(tr.stats.items()), expected)
        expected = np.array([895, 867, 747, 591, 359, -129, -185, 3, 115, 243],
                            dtype=np.int16)
        np.testing.assert_array_equal(tr.data[:10], expected)

    def test_is_pdas(self):
        """
        Tests the _read_pdas function.
        """
        self.assertTrue(_is_pdas(self.testfile))


def suite():
    return unittest.makeSuite(PDASTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
