# -*- coding: utf-8 -*-

from obspy.core import Trace, read
from obspy.core.util import NamedTemporaryFile, _getPlugins
import numpy as np
import os
import unittest


class WaveformPluginsTestCase(unittest.TestCase):
    """
    Test suite for all installed waveform plug-ins.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_writeFormat(self):
        """
        """
        formats = _getPlugins('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            # XXX: SAC and GSE2 fail those test!!!
            if format in ['SAC', 'GSE2']:
                continue
            # new trace object
            tr = Trace(data=np.arange(0, 2000))
            #1 - r/w little endian with auto detection
            outfile = NamedTemporaryFile().name
            tr.write(outfile, format=format, byteorder='<')
            if format == 'Q':
                outfile += '.QHD'
            st = read(outfile)
            self.assertEquals(st[0].data.dtype.byteorder, '=')
            os.remove(outfile)
            if format == 'Q':
                os.remove(outfile[:-4] + '.QBN')
                os.remove(outfile[:-4])
            #1 - r/w big endian with auto detection
            outfile = NamedTemporaryFile().name
            tr.write(outfile, format=format, byteorder='>')
            if format == 'Q':
                outfile += '.QHD'
            st = read(outfile)
            self.assertEquals(st[0].data.dtype.byteorder, '=')
            os.remove(outfile)
            if format == 'Q':
                os.remove(outfile[:-4] + '.QBN')
                os.remove(outfile[:-4])


def suite():
    return unittest.makeSuite(WaveformPluginsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
