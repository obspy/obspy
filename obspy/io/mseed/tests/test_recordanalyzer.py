#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy.core.util.misc import CatchOutput
from obspy.io.mseed.scripts.recordanalyzer import main as obspy_recordanalyzer


class RecordAnalyserTestCase(unittest.TestCase):
    def setUp(self):
        self.test_file = os.path.join(os.path.dirname(__file__),
                                      'data',
                                      'timingquality.mseed')

    def test_default_record(self):
        with CatchOutput() as out:
            obspy_recordanalyzer([self.test_file])

        expected = '''FILE: %s
Record Number: 0
Record Offset: 0 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
	Sequence number: 763445
	Data header/quality indicator: D
	Station identifier code: BGLD
	Location identifier: 
	Channel identifier: EHE
	Network code: BW
	Record start time: 2007-12-31T23:59:59.915000Z
	Number of samples: 412
	Sample rate factor: 200
	Sample rate multiplier: 1
	Activity flags: 0
	I/O and clock flags: 0
	Data quality flags: 0
	Number of blockettes that follow: 2
	Time correction: -1500
	Beginning of data: 64
	First blockette: 48

BLOCKETTES
	1000:	Encoding Format: 10
		Word Order: 1
		Data Record Length: 9
	1001:	Timing quality: 55
		mu_sec: 0
		Frame count: -73

CALCULATED VALUES
	Corrected Starttime: 2007-12-31T23:59:59.765000Z

''' % (self.test_file,)  # noqa
        self.assertEqual(expected.encode('utf-8'),  # noqa
                         out.stdout)

    def test_second_record(self):
        with CatchOutput() as out:
            obspy_recordanalyzer(['-n', '1', self.test_file])

        expected = '''FILE: %s
Record Number: 1
Record Offset: 512 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
	Sequence number: 763446
	Data header/quality indicator: D
	Station identifier code: BGLD
	Location identifier: 
	Channel identifier: EHE
	Network code: BW
	Record start time: 2008-01-01T00:00:01.975000Z
	Number of samples: 412
	Sample rate factor: 200
	Sample rate multiplier: 1
	Activity flags: 0
	I/O and clock flags: 0
	Data quality flags: 0
	Number of blockettes that follow: 2
	Time correction: -1500
	Beginning of data: 64
	First blockette: 48

BLOCKETTES
	1000:	Encoding Format: 10
		Word Order: 1
		Data Record Length: 9
	1001:	Timing quality: 70
		mu_sec: 0
		Frame count: -73

CALCULATED VALUES
	Corrected Starttime: 2008-01-01T00:00:01.825000Z

''' % (self.test_file,)  # noqa
        self.assertEqual(expected.encode('utf-8'),  # noqa
                         out.stdout)

    def test_record_with_data_offset_zero(self):
        """
        The test file has a middle record which has data offset zero. Make
        sure it can be read, as well as the following record.
        """
        filename = os.path.join(os.path.dirname(__file__), 'data', 'bizarre',
                                'mseed_data_offset_0.mseed')

        with CatchOutput() as out:
            obspy_recordanalyzer(['-n', '1', filename])

        expected = '''FILE: %s
Record Number: 1
Record Offset: 512 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
	Sequence number: 1
	Data header/quality indicator: D
	Station identifier code: PANIX
	Location identifier: 
	Channel identifier: LHZ
	Network code: CH
	Record start time: 2016-08-21T01:43:37.000000Z
	Number of samples: 0
	Sample rate factor: 1
	Sample rate multiplier: 1
	Activity flags: 4
	I/O and clock flags: 0
	Data quality flags: 0
	Number of blockettes that follow: 2
	Time correction: 0
	Beginning of data: 0
	First blockette: 48

BLOCKETTES
	1000:	Encoding Format: 11
		Word Order: 1
		Data Record Length: 9
	201:	NOT YET IMPLEMENTED

CALCULATED VALUES
	Corrected Starttime: 2016-08-21T01:43:37.000000Z

''' % (filename,)  # noqa
        self.assertEqual(expected.encode('utf-8'),  # noqa
                         out.stdout)

        with CatchOutput() as out:
            obspy_recordanalyzer(['-n', '2', filename])

        expected = '''FILE: %s
Record Number: 2
Record Offset: 1024 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
	Sequence number: 189
	Data header/quality indicator: D
	Station identifier code: PANIX
	Location identifier: 
	Channel identifier: LHZ
	Network code: CH
	Record start time: 2016-08-21T01:45:31.000000Z
	Number of samples: 262
	Sample rate factor: 1
	Sample rate multiplier: 1
	Activity flags: 68
	I/O and clock flags: 32
	Data quality flags: 0
	Number of blockettes that follow: 2
	Time correction: 0
	Beginning of data: 64
	First blockette: 48

BLOCKETTES
	1000:	Encoding Format: 11
		Word Order: 1
		Data Record Length: 9
	1001:	Timing quality: 100
		mu_sec: 0
		Frame count: 7

CALCULATED VALUES
	Corrected Starttime: 2016-08-21T01:45:31.000000Z

''' % (filename,)  # noqa
        self.assertEqual(expected.encode('utf-8'),  # noqa
                         out.stdout)


def suite():
    return unittest.makeSuite(RecordAnalyserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
