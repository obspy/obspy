#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy.core.util.misc import CatchOutput
from obspy.mseed.scripts.recordanalyzer import main as obspy_recordanalyzer


class RecordAnalyserTestCase(unittest.TestCase):
    def setUp(self):
        self.test_file = os.path.join(os.path.dirname(__file__),
                                      'data',
                                      'test.mseed')

    def test_default_record(self):
        with CatchOutput() as out:
            obspy_recordanalyzer([self.test_file])

        expected = '''FILE: %s
Record Number: 0
Record Offset: 0 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
	Sequence number: 1
	Data header/quality indicator: R
	Station identifier code: HGN
	Location identifier: 00
	Channel identifier: BHZ
	Network code: NL
	Record start time: 2003-05-29T02:13:22.043400Z
	Number of samples: 5980
	Sample rate factor: 32760
	Sample rate multiplier: 64717
	Activity flags: 0
	I/O and clock flags: 0
	Data quality flags: 0
	Number of blockettes that follow: 2
	Time correction: 0
	Beginning of data: 128
	First blockette: 48

BLOCKETTES
	1000:	Encoding Format: 11
		Word Order: 1
		Data Record Length: 12
	100:	Sampling Rate: 40.0

CALCULATED VALUES
	Corrected Starttime: 2003-05-29T02:13:22.043400Z

''' % (self.test_file,)  # noqa
        self.assertEqual(expected.encode('utf-8'),
                         out.stdout)

    def test_second_record(self):
        with CatchOutput() as out:
            obspy_recordanalyzer(['-n', '1', self.test_file])

        expected = '''FILE: %s
Record Number: 1
Record Offset: 4096 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
	Sequence number: 2
	Data header/quality indicator: R
	Station identifier code: HGN
	Location identifier: 00
	Channel identifier: BHZ
	Network code: NL
	Record start time: 2003-05-29T02:15:51.543400Z
	Number of samples: 5967
	Sample rate factor: 32760
	Sample rate multiplier: 64717
	Activity flags: 0
	I/O and clock flags: 0
	Data quality flags: 0
	Number of blockettes that follow: 2
	Time correction: 0
	Beginning of data: 128
	First blockette: 48

BLOCKETTES
	1000:	Encoding Format: 11
		Word Order: 1
		Data Record Length: 12
	100:	Sampling Rate: 40.0

CALCULATED VALUES
	Corrected Starttime: 2003-05-29T02:15:51.543400Z

''' % (self.test_file,)  # noqa
        self.assertEqual(expected.encode('utf-8'),
                         out.stdout)


def suite():
    return unittest.makeSuite(RecordAnalyserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
