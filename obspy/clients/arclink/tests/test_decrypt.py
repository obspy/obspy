# -*- coding: utf-8 -*-
"""
The obspy.clients.arclink.client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy.clients.arclink import Client, decrypt
from obspy.clients.arclink.client import DCID_KEY_FILE, ArcLinkException
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile


@unittest.skipIf(not decrypt.HAS_CRYPTOLIB,
                 'M2Crypto, PyCrypto or cryptography is not installed')
class ClientTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.clients.arclink.client.Client}.
    """
    def test_get_waveform_with_dcid_key(self):
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
        stream1 = client1.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
        # compare results
        np.testing.assert_array_equal(stream1[0].data, stream2[0].data)
        self.assertEqual(stream1[0].stats, stream2[0].stats)

    def test_get_waveform_with_dcid_key_file(self):
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
        stream1 = client1.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
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
        stream1 = client1.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
        # compare results
        np.testing.assert_array_equal(stream1[0].data, stream2[0].data)
        self.assertEqual(stream1[0].stats, stream2[0].stats)

    @unittest.skipIf(os.path.isfile(DCID_KEY_FILE),
                     '$HOME/dcidpasswords.txt already exists')
    def test_get_waveform_with_default_dcid_key_file(self):
        """
        Use $HOME/dcidpasswords.txt.
        """
        dcidfile = DCID_KEY_FILE
        fh = open(dcidfile, 'wt')
        fh.write('TEST=XYZ\r\nBIA=OfH9ekhi\r\n')
        fh.close()
        try:
            # test server for encryption
            client1 = Client(
                host="webdc.eu", port=36000, user="test@obspy.org")
            # public server
            client2 = Client(
                host="webdc.eu", port=18001, user="test@obspy.org")
        finally:
            # clean up dcid file
            os.remove(dcidfile)
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        stream1 = client1.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
        stream2 = client2.get_waveforms('GE', 'APE', '', 'BHZ', start, end)
        # compare results
        np.testing.assert_array_equal(stream1[0].data, stream2[0].data)
        self.assertEqual(stream1[0].stats, stream2[0].stats)

    def test_get_waveform_unknown_user(self):
        """
        Unknown user raises an ArcLinkException: DENIED.
        """
        client = Client(host="webdc.eu", port=36000, user="unknown@obspy.org")
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        self.assertRaises(ArcLinkException, client.get_waveforms, 'GE', 'APE',
                          '', 'BHZ', start, end)

    def test_get_waveform_wrong_password(self):
        """
        A wrong password password raises exception.
        """
        client = Client(host="webdc.eu", port=36000, user="test@obspy.org",
                        dcid_keys={'BIA': 'WrongPassword'})
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        self.assertRaises(Exception, client.get_waveforms,
                          'GE', 'APE', '', 'BHZ', start, end)

    def test_get_waveform_no_password(self):
        """
        No password raises exception.
        """
        client = Client(host="webdc.eu", port=36000, user="test@obspy.org",
                        dcid_keys={'BIA': ''})
        # request data
        start = UTCDateTime(2010, 1, 1, 10, 0, 0)
        end = start + 100
        self.assertRaises(Exception, client.get_waveforms,
                          'GE', 'APE', '', 'BHZ', start, end)

    @unittest.skipIf(not decrypt.HAS_CRYPTOGRAPHY,
                     'cryptography is not installed')
    def test_cryptography(self):
        """
        Test cryptography by temporarly disabling all other crypto libs
        """
        # monkey patch
        backup = decrypt.HAS_M2CRYPTO, decrypt.HAS_PYCRYPTO
        decrypt.HAS_M2CRYPTO = False
        decrypt.HAS_PYCRYPTO = False
        # run test
        self.test_get_waveform_with_dcid_key()
        # revert monkey patch
        decrypt.HAS_M2CRYPTO, decrypt.HAS_PYCRYPTO = backup

    @unittest.skipIf(not decrypt.HAS_PYCRYPTO, 'PyCrypto is not installed')
    def test_pycrypto(self):
        """
        Test PyCrypto by temporarly disabling all other crypto libs
        """
        # monkey patch
        backup = decrypt.HAS_M2CRYPTO, decrypt.HAS_CRYPTOGRAPHY
        decrypt.HAS_M2CRYPTO = False
        decrypt.HAS_CRYPTOGRAPHY = False
        # run test
        self.test_get_waveform_with_dcid_key()
        # revert monkey patch
        decrypt.HAS_M2CRYPTO, decrypt.HAS_CRYPTOGRAPHY = backup

    @unittest.skipIf(not decrypt.HAS_M2CRYPTO, 'M2Crypto is not installed')
    def test_m2crypto(self):
        """
        Test M2Crypto by temporarly disabling all other crypto libs
        """
        # monkey patch
        backup = decrypt.HAS_CRYPTOGRAPHY, decrypt.HAS_PYCRYPTO
        decrypt.HAS_CRYPTOGRAPHY = False
        decrypt.HAS_PYCRYPTO = False
        # run test
        self.test_get_waveform_with_dcid_key()
        # revert monkey patch
        decrypt.HAS_CRYPTOGRAPHY, decrypt.HAS_PYCRYPTO = backup


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
