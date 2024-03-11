#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.multiclient test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import configparser
from unittest import mock

import pytest

from obspy import UTCDateTime
from obspy.core.util.base import NamedTemporaryFile
from obspy.clients import MultiClient


CONFIG = \
    """
    [lookup]
    US = fdsn_iris
    GR = fdsn_bgr
    GR.WET = sds1

    [fdsn_iris]
    type = fdsn
    base_url = IRIS
    user_agent = LMU
    timeout = 30
    debug = false

    [fdsn_bgr]
    type = fdsn
    base_url = http://eida.bgr.de
    user_agent = LMU
    timeout = 20

    [sds1]
    type = sds
    sds_root = /path/to/SDS/archive
    sds_type = D
    format = MSEED
    fileborder_seconds = 30
    fileborder_samples = 999
    FMTSTR = {year}/{doy:03d}/*.mseed
    """
t1 = UTCDateTime('2023-05-20T12:34:56')
t2 = t1 + 10


class TestMultiClient():
    """
    Test case for obspy.clients.multiclient.MultiClient

    Mock out anything that would go over network, just test that the expected
    submodule client initializations (fdsn, sds) are made and that the expected
    get_waveforms() calls go out to these mocked instances
    """
    def test_multiclient(self):
        """
        Tests the MultiClient as a whole since the __init__ is just parsing the
        config file and then all intersting stuff is happening directly in
        consecutive `get_waveform(..)` calls.
        """
        with NamedTemporaryFile() as tf:
            filename = tf.name
            with open(filename, 'wt', encoding="ASCII") as fh:
                fh.write(CONFIG)

            client = MultiClient(filename)

        # mock out FDSN and SDS Client
        # this is a bit ugly.. in principle it whould work with decorators on
        # the test method but somehow could not get it to work, it seems to be
        # tricky to catch the right namespace and the "where to patch" section
        # in the mock.patch docs did not help as much
        #      @mock.patch('obspy.clients.filesystem.sds.Client')
        # to work around it just put mock objects where in real life the
        # original obspy Client classes for FDSN and SDS would be stored
        mock_sds_client = mock.Mock()
        mock_fdsn_client = mock.Mock()
        client.supported_client_types['sds'] = mock_sds_client
        client.supported_client_types['fdsn'] = mock_fdsn_client

        assert client._config.sections() == [
            'lookup', 'fdsn_iris', 'fdsn_bgr', 'sds1']
        assert client._config.options('lookup') == ['US', 'GR', 'GR.WET']
        assert client._config.options('fdsn_iris') == [
            'type', 'base_url', 'user_agent', 'timeout', 'debug']
        assert client._config.items('sds1') == [
            ('type', 'sds'), ('sds_root', '/path/to/SDS/archive'),
            ('sds_type', 'D'), ('format', 'MSEED'),
            ('fileborder_seconds', '30'), ('fileborder_samples', '999'),
            ('FMTSTR', '{year}/{doy:03d}/*.mseed')]

        # at this point no client connections should have been made
        assert client._clients == {}
        assert not mock_fdsn_client.called
        assert not mock_sds_client.called

        # request data that should come from FDSN IRIS according to lookup
        # config
        client.get_waveforms("US", "XYZ", "*", "HH?", t1, t2)
        # fdsn client init should now have been called exactly once to
        # initialize IRIS FDSNWS
        mock_fdsn_client.assert_called_once_with(
            base_url='IRIS', user_agent='LMU', debug=False, timeout=30.0)
        mock_fdsn = client._clients['fdsn_iris']
        assert len(client._clients) == 1
        mock_fdsn.get_waveforms.assert_called_once_with(
            'US', 'XYZ', '*', 'HH?', t1, t2)
        # do another request, should not make a new client connection
        client.get_waveforms("US", "ABC", "", "BH?", t2, t1)
        assert mock_fdsn is client._clients['fdsn_iris']
        mock_fdsn_client.assert_called_once()
        assert len(client._clients) == 1
        assert mock_fdsn.get_waveforms.call_count == 2
        mock_fdsn.get_waveforms.assert_called_with(
            'US', 'ABC', '', 'BH?', t2, t1)

        # now do a request that simulates another FDSNWS connection to BGR
        client.get_waveforms("GR", "FUR", "*", "HH?", t1, t2)
        # this initializes a new client..
        # (but the mock client class always returns the same child mock that it
        # originally created, so we are testing the same mock client connection
        # object still)
        assert len(client._clients) == 2
        assert client._clients == {
            'fdsn_iris': mock_fdsn, 'fdsn_bgr': mock_fdsn}
        assert mock_fdsn_client.call_count == 2
        mock_fdsn_client.assert_called_with(
            base_url='http://eida.bgr.de', user_agent='LMU', timeout=20.0)
        # check that the get_waveforms call was made
        # again, due to how Mock.return_value works these calls are made on the
        # same client "instance" as IRIS above, in reality they would have been
        # made on a totally different client instance object
        assert mock_fdsn.get_waveforms.call_count == 3
        mock_fdsn.get_waveforms.assert_called_with(
            "GR", "FUR", "*", "HH?", t1, t2)

        # now make a request for GR.WET which according to lookup config should
        # not go to BGR but instead is defined for a local SDS filesystem
        # client
        assert not mock_sds_client.called
        client.get_waveforms("GR", "WET", "00", "B*", t1, t2)
        mock_sds_client.assert_called_once_with(
            sds_root='/path/to/SDS/archive', sds_type='D', format='MSEED',
            fileborder_seconds=30.0, fileborder_samples=999)
        mock_sds = client._clients['sds1']
        assert len(client._clients) == 3
        assert client._clients == {
            'fdsn_iris': mock_fdsn, 'fdsn_bgr': mock_fdsn, 'sds1': mock_sds}
        mock_sds.get_waveforms.assert_called_once_with(
            "GR", "WET", "00", "B*", t1, t2)

        # now make a get_waveform call that can not be resolved as the lookup
        # config contains no info for this station/network
        msg = r"Found no matching lookup keys for requested data 'AA\.BCD'"
        with pytest.raises(configparser.Error, match=msg):
            client.get_waveforms('AA', 'BCD', '', '*', t1, t2)
