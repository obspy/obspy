# -*- coding: utf-8 -*-

from obspy.core import Trace, read
from obspy.core.utcdatetime import UTCDateTime
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
        STARTTIME = UTCDateTime(2009, 1, 13, 12, 1, 2, 999000)
        formats = _getPlugins('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            for byteorder in ['<', '>', '=']:
                # XXX: SAC and GSE2 fail those test!!!
                if format in ['SAC', 'GSE2']:
                    continue
                # new trace object
                tr = Trace(data=np.arange(0, 2000))
                tr.stats.network = "BW"
                tr.stats.station = "MANZ1"
                tr.stats.location = "00"
                tr.stats.channel = "EHE"
                tr.stats.calib = 0.999999
                tr.stats.delta = 0.005
                tr.stats.starttime = STARTTIME
                # create waveform file with given format and byte order
                outfile = NamedTemporaryFile().name
                tr.write(outfile, format=format, byteorder=byteorder)
                if format == 'Q':
                    outfile += '.QHD'
                # read in again using auto detection
                st = read(outfile)
                self.assertEquals(len(st), 1)
                self.assertEquals(st[0].stats._format, format)
                # read in using format argument
                st = read(outfile, format=format)
                self.assertEquals(len(st), 1)
                self.assertEquals(st[0].stats._format, format)
                # check byte order
                self.assertEquals(st[0].data.dtype.byteorder, '=')
                # check meta data
                if format not in ['MSEED', 'WAV']:
                    # MSEED does not contain the calibration factor
                    self.assertAlmostEquals(st[0].stats.calib, 0.999999)
                if format not in ['WAV']:
                    self.assertEquals(st[0].stats.starttime, STARTTIME)
                    self.assertEquals(st[0].stats.endtime, STARTTIME + 9.995)
                    self.assertEquals(st[0].stats.delta, 0.005)
                    self.assertEquals(st[0].stats.sampling_rate, 200.0)
                # network/station/location/channel codes
                if format in ['Q', 'SH_ASC']:
                    # no network or location code in Q or SH_ASC
                    self.assertEquals(st[0].id, ".MANZ1..EHE")
                elif format not in ['WAV']:
                    self.assertEquals(st[0].id, "BW.MANZ1.00.EHE")
                # remove temporary files
                os.remove(outfile)
                if format == 'Q':
                    os.remove(outfile[:-4] + '.QBN')
                    os.remove(outfile[:-4])


def suite():
    return unittest.makeSuite(WaveformPluginsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
