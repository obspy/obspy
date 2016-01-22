# -*- coding: utf-8 -*-
"""
The obspy.clients.syngine test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import io
import unittest

import numpy as np

import obspy
from obspy.core.compatibility import mock
from obspy.core.util.misc import CatchOutput
from obspy.clients.syngine import Client
from obspy.clients.base import DEFAULT_TESTING_USER_AGENT

BASE_URL = "http://service.iris.edu/irisws/syngine/1"


class RequestsMockResponse(object):
    def __init__(self):
        self.text = ""
        self.content = b""
        self.status_code = 200
        self._json = {}

    def json(self):
        return self._json


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.iris.client.Client.
    """
    c = Client(user_agent=DEFAULT_TESTING_USER_AGENT)

    def test_get_model_info_mock(self):
        """
        Mock test for the get_model_info() method.
        """
        with mock.patch("requests.get") as p:
            r = RequestsMockResponse()
            r._json["slip"] = [0.0, 1.0, 2.0]
            r._json["sliprate"] = [0.0, 1.0, 2.0]
            p.return_value = r
            self.c.get_model_info("test_model")

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         'http://service.iris.edu/irisws/syngine/1/info')
        self.assertEqual(p.call_args[1]["params"],
                         {'model': 'test_model'})
        self.assertEqual(p.call_args[1]["headers"],
                         {'User-Agent': DEFAULT_TESTING_USER_AGENT})

    def test_get_model_info(self):
        """
        Actual test for the get_model_info() method.
        """
        info = self.c.get_model_info("test")

        self.assertTrue(isinstance(info, obspy.core.AttribDict))
        # Check two random keys.
        self.assertEqual(info.dump_type, "displ_only")
        self.assertEqual(info.time_scheme, "newmark2")
        # Check that both arrays have been converted to numpy arrays.
        self.assertTrue(isinstance(info.slip, np.ndarray))
        self.assertTrue(isinstance(info.sliprate, np.ndarray))

    def test_get_available_models_mock(self):
        with mock.patch("requests.get") as p:
            p.return_value = RequestsMockResponse()
            self.c.get_available_models()

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         'http://service.iris.edu/irisws/syngine/1/models')
        self.assertEqual(p.call_args[1]["params"], None)
        self.assertEqual(p.call_args[1]["headers"],
                         {'User-Agent': DEFAULT_TESTING_USER_AGENT})

    def test_get_available_models(self):
        models = self.c.get_available_models()
        self.assertTrue(isinstance(models, dict))
        keys = list(models.keys())
        self.assertTrue(len(keys) > 3)
        self.assertIn("ak135f_5s", keys)
        # Check random key.
        self.assertEqual(models["ak135f_5s"]["components"],
                         "vertical and horizontal")

    def test_print_model_information(self):
        with mock.patch("requests.get") as p:
            p.return_value = RequestsMockResponse()
            p.return_value._json = {"a": "b"}

            with CatchOutput() as out:
                self.c.print_model_information()

        self.assertEqual(out.stdout, b"{'a': 'b'}\n")

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         'http://service.iris.edu/irisws/syngine/1/models')
        self.assertEqual(p.call_args[1]["params"], None)
        self.assertEqual(p.call_args[1]["headers"],
                         {'User-Agent': DEFAULT_TESTING_USER_AGENT})

    def test_get_service_version_mock(self):
        with mock.patch("requests.get") as p:
            p.return_value = RequestsMockResponse()
            p.return_value.text = "1.2.3"
            version = self.c.get_service_version()

        self.assertEqual(version, "1.2.3")

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         'http://service.iris.edu/irisws/syngine/1/version')
        self.assertEqual(p.call_args[1]["params"], None)
        self.assertEqual(p.call_args[1]["headers"],
                         {'User-Agent': DEFAULT_TESTING_USER_AGENT})

    def test_get_waveforms_mock(self):
        """
        Test the queries from the IRIS syngine website and see if they
        produce the correct URLS.
        """
        r = RequestsMockResponse()
        with io.BytesIO() as buf:
            obspy.read()[0].write(buf, format="mseed")
            buf.seek(0, 0)
            r.content = buf.read()

        # http://service.iris.edu/irisws/syngine/1/query?network=IU&
        # station=ANMO&components=ZRT&eventid=GCMT:M110302J
        with mock.patch("requests.get") as p:
            p.return_value = r
            st = self.c.get_waveforms(model="ak135f_5s",
                                      network="IU", station="ANMO",
                                      components="ZRT",
                                      eventid="GCMT:M110302J")

        self.assertTrue(isinstance(st, obspy.Stream))

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         "http://service.iris.edu/irisws/syngine/1/query")
        self.assertEqual(p.call_args[1]["params"], {
            "components": "ZRT",
            "eventid": "GCMT:M110302J",
            "format": "miniseed",
            "model": "ak135f_5s",
            "network": "IU",
            "station": "ANMO"})
        self.assertEqual(p.call_args[1]["headers"],
                         {"User-Agent": DEFAULT_TESTING_USER_AGENT})

        # http://service.iris.edu/irisws/syngine/1/query?network=_GSN&
        # components=Z&eventid=GCMT:M110302J&endtime=1800
        with mock.patch("requests.get") as p:
            p.return_value = r
            st = self.c.get_waveforms(model="ak135f_5s",
                                      network="_GSN",
                                      components="Z",
                                      endtime=1800.0,
                                      eventid="GCMT:M110302J")

        self.assertTrue(isinstance(st, obspy.Stream))

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         "http://service.iris.edu/irisws/syngine/1/query")
        self.assertEqual(p.call_args[1]["params"], {
            "components": "Z",
            "endtime": 1800.0,
            "eventid": "GCMT:M110302J",
            "format": "miniseed",
            "model": "ak135f_5s",
            "network": "_GSN"})
        self.assertEqual(p.call_args[1]["headers"],
                         {"User-Agent": DEFAULT_TESTING_USER_AGENT})

        # http://service.iris.edu/irisws/syngine/1/query?network=_GSN&
        # components=Z&eventid=GCMT:M110302J&starttime=P-10&endtime=ScS%2B60
        with mock.patch("requests.get") as p:
            p.return_value = r
            st = self.c.get_waveforms(model="ak135f_5s",
                                      network="_GSN",
                                      components="Z",
                                      starttime="P-10",
                                      endtime="ScS+60",
                                      eventid="GCMT:M110302J")

        self.assertTrue(isinstance(st, obspy.Stream))

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         "http://service.iris.edu/irisws/syngine/1/query")
        self.assertEqual(p.call_args[1]["params"], {
            "components": "Z",
            "starttime": "P-10",
            "endtime": "ScS+60",
            "eventid": "GCMT:M110302J",
            "format": "miniseed",
            "model": "ak135f_5s",
            "network": "_GSN"})
        self.assertEqual(p.call_args[1]["headers"],
                         {"User-Agent": DEFAULT_TESTING_USER_AGENT})

    def test_error_handling_arguments(self):
        # Floating points value
        self.assertRaises(ValueError, self.c.get_waveforms, model="test",
                          receiverlatitude="a")
        # Int.
        self.assertRaises(ValueError, self.c.get_waveforms, model="test",
                          kernelwidth="a")
        # Time.
        self.assertRaises(TypeError, self.c.get_waveforms, model="test",
                          origintime="a")

    def test_source_mechanisms_mock(self):
        r = RequestsMockResponse()
        with io.BytesIO() as buf:
            obspy.read()[0].write(buf, format="mseed")
            buf.seek(0, 0)
            r.content = buf.read()

        with mock.patch("requests.get") as p:
            p.return_value = r
            self.c.get_waveforms(model="ak135f_5s",
                                 sourcemomenttensor=[1, 2, 3, 4, 5, 6])
        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         "http://service.iris.edu/irisws/syngine/1/query")
        self.assertEqual(p.call_args[1]["params"], {
            "model": "ak135f_5s",
            "format": "miniseed",
            "sourcemomenttensor": "1,2,3,4,5,6"})

        with mock.patch("requests.get") as p:
            p.return_value = r
            self.c.get_waveforms(model="ak135f_5s",
                                 sourcedoublecouple=[1, 2, 3, 4])
        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         "http://service.iris.edu/irisws/syngine/1/query")
        self.assertEqual(p.call_args[1]["params"], {
            "model": "ak135f_5s",
            "format": "miniseed",
            "sourcedoublecouple": "1,2,3,4"})

        with mock.patch("requests.get") as p:
            p.return_value = r
            self.c.get_waveforms(model="ak135f_5s",
                                 sourceforce=[3.32, 4.23, 5.11])
        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[1]["url"],
                         "http://service.iris.edu/irisws/syngine/1/query")
        self.assertEqual(p.call_args[1]["params"], {
            "model": "ak135f_5s",
            "format": "miniseed",
            "sourceforce": "3.32,4.23,5.11"})


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
