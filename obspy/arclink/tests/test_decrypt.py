# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy.arclink import Client
from obspy.arclink.client import DCID_KEY_FILE, ArcLinkException
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile, skipIf

try:
    from M2Crypto.EVP import EVPError
    hasM2Crypto = True
except ImportError:
    hasM2Crypto = False


class ClientTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.arclink.client.Client}.
    """
    @skipIf(not hasM2Crypto, 'Module M2Crypto is not installed')
    def test_getWaveformWithDCIDKey(self):
        """
        """
        # test server for encryption
        client1 = Client(host="webdc.eu", port=36000, user="test@obspy.org",
                         dcid_keys={'BIA': 'OfH9ekhi'})
        # public server
        client2 = Client(host="webdc.eu", port=18001, user="test@obspy.org")
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        stream1 = client1.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        # compare results
        np.testing.assert_array_equal(stream1[0].data, stream2[0].data)
        self.assertEqual(stream1[0].stats, stream2[0].stats)

    @skipIf(not hasM2Crypto, 'Module M2Crypto is not installed')
    def test_getWaveformWithDCIDKeyFile(self):
        """
        Tests various DCID key file formats (with space or equal sign). Also
        checks if empty lines or comment lines are ignored.
        """
        # 1 - using = sign between username and password
        with NamedTemporaryFile() as tf:
            dcidfile = tf.name
            with open(dcidfile, 'wt') as fh:
                fh.write('#Comment\n\n\nTEST=XYZ\r\nBIA=OfH9ekhi\r\n')
            # test server for encryption
            client1 = Client(host="webdc.eu", port=36000,
                             user="test@obspy.org", dcid_key_file=dcidfile)
            # public server
            client2 = Client(host="webdc.eu", port=18001,
                             user="test@obspy.org")
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        stream1 = client1.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        # compare results
        np.testing.assert_array_equal(stream1[0].data, stream2[0].data)
        self.assertEqual(stream1[0].stats, stream2[0].stats)
        # 2 - using space between username and password
        with NamedTemporaryFile() as tf:
            dcidfile = tf.name
            with open(dcidfile, 'wt') as fh:
                fh.write('TEST XYZ\r\nBIA OfH9ekhi\r\n')
            # test server for encryption
            client1 = Client(host="webdc.eu", port=36000,
                             user="test@obspy.org", dcid_key_file=dcidfile)
            # public server
            client2 = Client(host="webdc.eu", port=18001,
                             user="test@obspy.org")
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        stream1 = client1.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        # compare results
        np.testing.assert_array_equal(stream1[0].data, stream2[0].data)
        self.assertEqual(stream1[0].stats, stream2[0].stats)

    @skipIf(os.path.isfile(DCID_KEY_FILE),
            '$HOME/dcidpasswords.txt already exists')
    @skipIf(not hasM2Crypto, 'Module M2Crypto is not installed')
    def test_getWaveformWithDefaultDCIDKeyFile(self):
        """
        Use $HOME/dcidpasswords.txt.
        """
        dcidfile = DCID_KEY_FILE
        fh = open(dcidfile, 'wt')
        fh.write('TEST=XYZ\r\nBIA=OfH9ekhi\r\n')
        fh.close()
        # test server for encryption
        client1 = Client(host="webdc.eu", port=36000, user="test@obspy.org")
        # public server
        client2 = Client(host="webdc.eu", port=18001, user="test@obspy.org")
        # clean up dcid file
        os.remove(dcidfile)
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        stream1 = client1.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.getWaveform('GE', 'APE', '', 'BHZ', start, end)
        # compare results
        np.testing.assert_array_equal(stream1[0].data, stream2[0].data)
        self.assertEqual(stream1[0].stats, stream2[0].stats)

    @skipIf(not hasM2Crypto, 'Module M2Crypto is not installed')
    def test_getWaveformUnknownUser(self):
        """
        Unknown user raises an ArcLinkException: DENIED.
        """
        client = Client(host="webdc.eu", port=36000, user="unknown@obspy.org")
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        self.assertRaises(ArcLinkException, client.getWaveform, 'GE', 'APE',
                          '', 'BHZ', start, end)

    @skipIf(not hasM2Crypto, 'Module M2Crypto is not installed')
    def test_getWaveformWrongPassword(self):
        """
        A wrong password password raises a "EVPError: bad decrypt".
        """
        client = Client(host="webdc.eu", port=36000, user="test@obspy.org",
                        dcid_keys={'BIA': 'WrongPassword'})
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        self.assertRaises(EVPError, client.getWaveform, 'GE', 'APE', '', 'BHZ',
                          start, end)

    @skipIf(not hasM2Crypto, 'Module M2Crypto is not installed')
    def test_getWaveformNoPassword(self):
        """
        No password raises a "EVPError: bad decrypt".
        """
        client = Client(host="webdc.eu", port=36000, user="test@obspy.org",
                        dcid_keys={'BIA': ''})
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        self.assertRaises(EVPError, client.getWaveform, 'GE', 'APE', '', 'BHZ',
                          start, end)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
