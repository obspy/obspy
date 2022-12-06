# -*- coding: utf-8 -*-
"""
The obspy.clients.syngine test suite.
"""
import io
import unittest
from unittest import mock

import numpy as np
import pytest

import obspy
from obspy.core.util.base import NamedTemporaryFile
from obspy.clients.syngine import Client
from obspy.clients.base import DEFAULT_TESTING_USER_AGENT, ClientHTTPException


BASE_URL = "http://service.iris.edu/irisws/syngine/1"
pytestmark = pytest.mark.network


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

        self.assertIsInstance(info, obspy.core.AttribDict)
        # Check two random keys.
        self.assertEqual(info.dump_type, "displ_only")
        self.assertEqual(info.time_scheme, "newmark2")
        # Check that both arrays have been converted to numpy arrays.
        self.assertIsInstance(info.slip, np.ndarray)
        self.assertIsInstance(info.sliprate, np.ndarray)

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
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 3)
        self.assertIn("ak135f_5s", models)
        # Check random key.
        self.assertEqual(models["ak135f_5s"]["components"],
                         "vertical and horizontal")

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

        self.assertIsInstance(st, obspy.Stream)

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

        self.assertIsInstance(st, obspy.Stream)

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

        self.assertIsInstance(st, obspy.Stream)

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
        with self.assertRaises(ValueError):
            self.c.get_waveforms(model="test", receiverlatitude="a")
        # Int.
        with self.assertRaises(ValueError):
            self.c.get_waveforms(model="test", kernelwidth="a")
        # Time.
        with self.assertRaises(TypeError):
            self.c.get_waveforms(model="test", origintime="a")

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

    def test_error_handling(self):
        """
        Tests the error handling. The clients just pass on most things to
        syngine and rely on the service for the error detection.
        """
        # Wrong components.
        with self.assertRaises(ClientHTTPException) as cm:
            self.c.get_waveforms(
                model="ak135f_5s", eventid="GCMT:C201002270634A",
                station="ANMO", network="IU", components="ABC")

        msg = cm.exception.args[0]
        self.assertIn("HTTP code 400 when", msg)
        self.assertIn("Unrecognized component", msg)

    def test_bulk_waveform_download_mock(self):
        """
        Mock the bulk download requests to test the payload generation.
        """
        r = RequestsMockResponse()
        with io.BytesIO() as buf:
            obspy.read()[0].write(buf, format="mseed")
            buf.seek(0, 0)
            r.content = buf.read()

        payload = []

        def side_effect(*args, **kwargs):
            payload[:] = [kwargs["data"].decode()]
            return r

        # Test simple lists first.
        with mock.patch("requests.post") as p:
            p.side_effect = side_effect
            self.c.get_waveforms_bulk(
                model="ak135f_5s", bulk=[
                    [1.0, 2.0],
                    (2.0, 3.0),
                    ("AA", "BB")])

        self.assertEqual(payload[0], "\n".join([
            "model=ak135f_5s",
            "format=miniseed",
            "1.0 2.0",
            "2.0 3.0",
            "AA BB\n"]))

        # A couple more parameters
        with mock.patch("requests.post") as p:
            p.side_effect = side_effect
            self.c.get_waveforms_bulk(
                model="ak135f_5s", bulk=[
                    [1.0, 2.0],
                    (2.0, 3.0),
                    ("AA", "BB")],
                format="miniseed",
                sourcemomenttensor=[1, 2, 3, 4, 5, 6])

        self.assertEqual(payload[0], "\n".join([
            "model=ak135f_5s",
            "format=miniseed",
            "sourcemomenttensor=1,2,3,4,5,6",
            "1.0 2.0",
            "2.0 3.0",
            "AA BB\n"]))

        # A couple of dictionaries.
        with mock.patch("requests.post") as p:
            p.side_effect = side_effect
            self.c.get_waveforms_bulk(
                model="ak135f_5s", bulk=[
                    {"network": "IU", "station": "ANMO"},
                    {"latitude": 12, "longitude": 13.1},
                    {"latitude": 12, "longitude": 13.1, "networkcode": "IU"},
                    {"latitude": 12, "longitude": 13.1, "stationcode": "ANMO"},
                    {"latitude": 12, "longitude": 13.1, "locationcode": "00"},
                    {"latitude": 12, "longitude": 13.1, "networkcode": "IU",
                     "stationcode": "ANMO", "locationcode": "00"}],
                format="miniseed", eventid="GCMT:C201002270634A")

        self.assertEqual(payload[0], "\n".join([
            "model=ak135f_5s",
            "eventid=GCMT:C201002270634A",
            "format=miniseed",
            "IU ANMO",
            "12 13.1",
            "12 13.1 NETCODE=IU",
            "12 13.1 STACODE=ANMO",
            "12 13.1 LOCCODE=00",
            "12 13.1 NETCODE=IU STACODE=ANMO LOCCODE=00\n"]))

    def test_get_waveforms(self):
        """
        Test get_waveforms() and get_waveforms_bulk() by actually downloading
        some things.

        Use the 'test' model which does not produce useful seismograms but
        is quick to test.
        """
        st = self.c.get_waveforms(model="test", network="IU", station="ANMO",
                                  eventid="GCMT:C201002270634A",
                                  components="Z")
        self.assertEqual(len(st), 1)
        # Download exactly the same with a bulk request and check the result
        # is the same!
        st_bulk = self.c.get_waveforms_bulk(
            model="test", bulk=[("IU", "ANMO")],
            eventid="GCMT:C201002270634A", components="Z")
        self.assertEqual(len(st_bulk), 1)
        self.assertEqual(st, st_bulk)

        # Test phase relative times. This tests that everything is correctly
        # encoded and what not.
        st = self.c.get_waveforms(model="test", network="IU", station="ANMO",
                                  eventid="GCMT:C201002270634A",
                                  starttime="P-10", endtime="P+20",
                                  components="Z")
        self.assertEqual(len(st), 1)
        st_bulk = self.c.get_waveforms_bulk(
            model="test", bulk=[("IU", "ANMO")],
            starttime="P-10", endtime="P+20",
            eventid="GCMT:C201002270634A", components="Z")
        self.assertEqual(len(st_bulk), 1)
        self.assertEqual(st, st_bulk)

        # One to test a source mechanism
        st = self.c.get_waveforms(model="test", network="IU", station="ANMO",
                                  sourcemomenttensor=[1, 2, 3, 4, 5, 6],
                                  sourcelatitude=10, sourcelongitude=20,
                                  sourcedepthinmeters=100,
                                  components="Z")
        self.assertEqual(len(st), 1)
        st_bulk = self.c.get_waveforms_bulk(
            model="test", bulk=[("IU", "ANMO")],
            sourcemomenttensor=[1, 2, 3, 4, 5, 6],
            sourcelatitude=10, sourcelongitude=20,
            sourcedepthinmeters=100,
            components="Z")
        self.assertEqual(len(st_bulk), 1)
        self.assertEqual(st, st_bulk)

        # One more to test actual time values.
        st = self.c.get_waveforms(
            model="test", network="IU", station="ANMO",
            origintime=obspy.UTCDateTime(2015, 1, 2, 3, 0, 5),
            starttime=obspy.UTCDateTime(2015, 1, 2, 3, 4, 5),
            endtime=obspy.UTCDateTime(2015, 1, 2, 3, 10, 5),
            sourcemomenttensor=[1, 2, 3, 4, 5, 6],
            sourcelatitude=10, sourcelongitude=20,
            sourcedepthinmeters=100,
            components="Z")
        self.assertEqual(len(st), 1)
        st_bulk = self.c.get_waveforms_bulk(
            model="test", bulk=[("IU", "ANMO")],
            origintime=obspy.UTCDateTime(2015, 1, 2, 3, 0, 5),
            starttime=obspy.UTCDateTime(2015, 1, 2, 3, 4, 5),
            endtime=obspy.UTCDateTime(2015, 1, 2, 3, 10, 5),
            sourcemomenttensor=[1, 2, 3, 4, 5, 6],
            sourcelatitude=10, sourcelongitude=20,
            sourcedepthinmeters=100,
            components="Z")
        self.assertEqual(len(st_bulk), 1)
        self.assertEqual(st, st_bulk)

    def test_saving_directly_to_file(self):
        # Save to a filename.
        with NamedTemporaryFile() as tf:
            filename = tf.name
            st = self.c.get_waveforms(
                model="test", network="IU", station="ANMO",
                eventid="GCMT:C201002270634A", starttime="P-10",
                endtime="P+10", components="Z", filename=tf)
            # No return value.
            self.assertTrue(st is None)

            st = obspy.read(filename)
            self.assertEqual(len(st), 1)

        # Save to an open file-like object.
        with io.BytesIO() as buf:
            st = self.c.get_waveforms(
                model="test", network="IU", station="ANMO",
                eventid="GCMT:C201002270634A", starttime="P-10",
                endtime="P+10", components="Z", filename=buf)
            # No return value.
            self.assertTrue(st is None)

            buf.seek(0, 0)
            st = obspy.read(buf)
            self.assertEqual(len(st), 1)

    def test_reading_saczip_files(self):
        st = self.c.get_waveforms(
            model="test", network="IU", station="ANMO",
            eventid="GCMT:C201002270634A", starttime="P-10",
            endtime="P+10", components="Z", format="saczip")
        self.assertEqual(len(st), 1)
        # Same with bulk request.
        st_bulk = self.c.get_waveforms_bulk(
            model="test", bulk=[("IU", "ANMO")],
            eventid="GCMT:C201002270634A", starttime="P-10",
            endtime="P+10", components="Z", format="saczip")
        self.assertEqual(len(st_bulk), 1)

        self.assertEqual(st, st_bulk)

    def test_bulk_waveform_send_custom_payload(self):
        """
        The get_waveforms_bulk() method can send a custom payload.
        """
        r = RequestsMockResponse()
        with io.BytesIO() as buf:
            obspy.read()[0].write(buf, format="mseed")
            buf.seek(0, 0)
            r.content = buf.read()

        payload = []

        def side_effect(*args, **kwargs):
            payload[:] = [kwargs["data"]]
            return r

        # Test simple lists first.
        with mock.patch("requests.post") as p:
            p.side_effect = side_effect
            self.c.get_waveforms_bulk(
                model="ak135f_5s", bulk=[], data=b"1234\n5678")

        self.assertEqual(payload[0], b"1234\n5678")
