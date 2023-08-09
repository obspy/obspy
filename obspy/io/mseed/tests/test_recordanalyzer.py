#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

from obspy.core.util.misc import CatchOutput
from obspy.io.mseed.scripts.recordanalyzer import main as obspy_recordanalyzer


class RecordAnalyserTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
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
    1000:    Encoding Format: 10
        Word Order: 1
        Data Record Length: 9
    1001:    Timing quality: 55
        mu_sec: 0
        Frame count: -73

CALCULATED VALUES
    Corrected Starttime: 2007-12-31T23:59:59.765000Z

''' % (self.test_file,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noqa

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
    1000:    Encoding Format: 10
        Word Order: 1
        Data Record Length: 9
    1001:    Timing quality: 70
        mu_sec: 0
        Frame count: -73

CALCULATED VALUES
    Corrected Starttime: 2008-01-01T00:00:01.825000Z

''' % (self.test_file,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noqa

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
    1000:    Encoding Format: 11
        Word Order: 1
        Data Record Length: 9
    201:    NOT YET IMPLEMENTED

CALCULATED VALUES
    Corrected Starttime: 2016-08-21T01:43:37.000000Z

''' % (filename,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noqa

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
    1000:    Encoding Format: 11
        Word Order: 1
        Data Record Length: 9
    1001:    Timing quality: 100
        mu_sec: 0
        Frame count: 7

CALCULATED VALUES
    Corrected Starttime: 2016-08-21T01:45:31.000000Z

''' % (filename,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noqa

    def test_record_with_negative_sr_fact_and_mult(self):
        """
        Regression tests as there was an issue with the record analyzer for
        negative sampling rates and factors.
        """
        filename = os.path.join(
            os.path.dirname(__file__), 'data',
            'single_record_negative_sr_fact_and_mult.mseed')

        with CatchOutput() as out:
            obspy_recordanalyzer([filename])

        expected = '''FILE: %s
Record Number: 0
Record Offset: 0 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
    Sequence number: 4
    Data header/quality indicator: M
    Station identifier code: TNV
    Location identifier:
    Channel identifier: VHZ
    Network code: MN
    Record start time: 1991-02-21T23:50:00.430000Z
    Number of samples: 60
    Sample rate factor: -10
    Sample rate multiplier: -1
    Activity flags: 0
    I/O and clock flags: 0
    Data quality flags: 0
    Number of blockettes that follow: 1
    Time correction: 0
    Beginning of data: 64
    First blockette: 48

BLOCKETTES
    1000:    Encoding Format: 10
        Word Order: 1
        Data Record Length: 12

CALCULATED VALUES
    Corrected Starttime: 1991-02-21T23:50:00.430000Z

''' % (filename,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noqa

    def test_step_cal_blockette(self):
        """
        Test the step calibration blockette type 300.
        """
        filename = os.path.join(
            os.path.dirname(__file__), 'data',
            'blockette300.mseed')

        with CatchOutput() as out:
            obspy_recordanalyzer([filename])

        expected = '''FILE: %s
Record Number: 0
Record Offset: 0 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
    Sequence number: 36680
    Data header/quality indicator: M
    Station identifier code: KIEV
    Location identifier: 00
    Channel identifier: BHZ
    Network code: IU
    Record start time: 2018-02-13T22:43:59.019500Z
    Number of samples: 20
    Sample rate factor: 20
    Sample rate multiplier: 1
    Activity flags: 1
    I/O and clock flags: 32
    Data quality flags: 0
    Number of blockettes that follow: 3
    Time correction: 0
    Beginning of data: 128
    First blockette: 48

BLOCKETTES
    1000:    Encoding Format: 11
        Word Order: 1
        Data Record Length: 9
    1001:    Timing quality: 100
        mu_sec: 38
        Frame count: 6
    300:    Calibration Start Time: 2018-02-13T22:44:00.000000Z
        Number of Step Calibrations: 1
        Step Duration in Seconds: 0
        Interval Duration in Seconds: 900
        Calibration Signal Amplitude: -30.0
        Calibration Monitor Channel: EC0
        Calibration Reference Amplitude: 0.0
        Coupling: resistive
        Rolloff: 3DB@10Hz

CALCULATED VALUES
    Corrected Starttime: 2018-02-13T22:43:59.019538Z

''' % (filename,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noqa

    def test_sine_cal_blockette(self):
        """
        Test the step calibration blockette type 310.
        """
        filename = os.path.join(
            os.path.dirname(__file__), 'data',
            'blockette310.mseed')

        with CatchOutput() as out:
            obspy_recordanalyzer([filename])

        expected = '''FILE: %s
Record Number: 0
Record Offset: 0 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
    Sequence number: 2624
    Data header/quality indicator: M
    Station identifier code: KIEV
    Location identifier: 00
    Channel identifier: LHZ
    Network code: IU
    Record start time: 2018-02-13T20:01:45.069500Z
    Number of samples: 10
    Sample rate factor: 1
    Sample rate multiplier: 1
    Activity flags: 1
    I/O and clock flags: 32
    Data quality flags: 0
    Number of blockettes that follow: 3
    Time correction: 0
    Beginning of data: 128
    First blockette: 48

BLOCKETTES
    1000:    Encoding Format: 11
        Word Order: 1
        Data Record Length: 9
    1001:    Timing quality: 100
        mu_sec: 38
        Frame count: 6
    310:    Calibration Start Time: 2018-02-13T20:02:00.000000Z
        Calibration Duration in Seconds: 2400
        Period of Signal in Seconds: 0
        Calibration Signal Amplitude: -30.0
        Calibration Monitor Channel: EC0
        Calibration Reference Amplitude: 0.0
        Coupling: resistive
        Rolloff: 3DB@10Hz    

CALCULATED VALUES
    Corrected Starttime: 2018-02-13T20:01:45.069538Z

''' % (filename,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noq

    def test_random_cal_blockette(self):
        """
        Test the random calibration blockette type 320.
        """
        filename = os.path.join(
            os.path.dirname(__file__), 'data',
            'blockette320.mseed')

        with CatchOutput() as out:
            obspy_recordanalyzer([filename])

        expected = '''FILE: %s
Record Number: 0
Record Offset: 0 byte
Header Endianness: Big Endian

FIXED SECTION OF DATA HEADER
    Sequence number: 2712
    Data header/quality indicator: M
    Station identifier code: KIEV
    Location identifier: 00
    Channel identifier: LHZ
    Network code: IU
    Record start time: 2018-02-13T23:26:57.069500Z
    Number of samples: 1
    Sample rate factor: 1
    Sample rate multiplier: 1
    Activity flags: 1
    I/O and clock flags: 32
    Data quality flags: 0
    Number of blockettes that follow: 3
    Time correction: 0
    Beginning of data: 128
    First blockette: 48

BLOCKETTES
    1000:    Encoding Format: 11
        Word Order: 1
        Data Record Length: 9
    1001:    Timing quality: 100
        mu_sec: 38
        Frame count: 6
    320:    Calibration Start Time: 2018-02-13T23:27:00.000000Z
        Calibration Duration in Seconds: 14400
        Peak-To-Peak Amplitude: -24
        Calibration Monitor Channel: EC0
        Calibration Reference Amplitude: 0
        Coupling: resistive
        Rolloff: 3DB@10Hz    
        Noise Type: Telegraf

CALCULATED VALUES
    Corrected Starttime: 2018-02-13T23:26:57.069538Z

''' % (filename,)  # noqa
        self.assertEqual(expected, out.stdout.replace("\t", "    "))  # noq
