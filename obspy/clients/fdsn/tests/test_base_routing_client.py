#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections
import unittest
import warnings
from unittest import mock

import pytest

import obspy
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.fdsn.routing.routing_client import (
    BaseRoutingClient, RoutingClient)
from obspy.clients.fdsn.routing.eidaws_routing_client import (
    EIDAWSRoutingClient)
from obspy.clients.fdsn.routing.federator_routing_client import (
    FederatorRoutingClient)


pytestmark = pytest.mark.network
_DummyResponse = collections.namedtuple("_DummyResponse", ["content"])


class BaseRoutingClientTestCase(unittest.TestCase):
    def setUp(self):
        # Need to inherit to add two required method by the abstract base
        # class.
        class _DummyBaseRoutingClient(BaseRoutingClient):  # pragma: no cover
            def get_service_version(self):
                """
                Return a semantic version number of the remote service as a
                string.
                """
                return "0.0.0"

            def _handle_requests_http_error(self, r):
                raise NotImplementedError

        self._cls_object = _DummyBaseRoutingClient
        self._cls = ("obspy.clients.fdsn.routing.routing_client."
                     "BaseRoutingClient")

    def test_router_intialization_helper_function(self):
        c = RoutingClient("eida-routing")
        self.assertIsInstance(c, EIDAWSRoutingClient)

        c = RoutingClient("iris-federator")
        self.assertIsInstance(c, FederatorRoutingClient)

        with self.assertRaises(NotImplementedError) as e:
            RoutingClient("random")
        self.assertEqual(
            e.exception.args[0],
            "Routing type 'random' is not implemented. Available types: "
            "`iris-federator`, `eida-routing`")

    def test_expansion_of_include_and_exclude_providers(self):
        c = self._cls_object(
            include_providers=["IRIS", "http://example.com"],
            exclude_providers=["BGR", "http://example2.com"])
        self.assertEqual(
            c.include_providers, ["service.iris.edu", "example.com"])
        self.assertEqual(
            c.exclude_providers, ["eida.bgr.de", "example2.com"])

        # None are set.
        c = self._cls_object()
        self.assertEqual(c.include_providers, [])
        self.assertEqual(c.exclude_providers, [])

        # Single strings.
        c = self._cls_object(include_providers="IRIS",
                             exclude_providers="BGR")
        self.assertEqual(c.include_providers, ["service.iris.edu"])
        self.assertEqual(c.exclude_providers, ["eida.bgr.de"])

        c = self._cls_object(include_providers="http://example.com/path",
                             exclude_providers="http://example2.com")
        self.assertEqual(c.include_providers, ["example.com/path"])
        self.assertEqual(c.exclude_providers, ["example2.com"])

    def test_request_filtering(self):
        split = {
            # Note that this is HTTPS.
            "https://example.com": "1234",
            "http://example2.com": "1234",
            "http://example3.com": "1234",
            "http://service.iris.edu": "1234"
        }

        c = self._cls_object(include_providers=["IRIS", "http://example.com"])
        self.assertEqual(c._filter_requests(split), {
            "https://example.com": "1234",
            "http://service.iris.edu": "1234"
        })

        c = self._cls_object(exclude_providers=["IRIS", "http://example.com"])
        self.assertEqual(c._filter_requests(split), {
            "http://example2.com": "1234",
            "http://example3.com": "1234"
        })

        # Both filters are always applied - it might result in zero
        # remaining providers.
        c = self._cls_object(include_providers=["IRIS", "http://example.com"],
                             exclude_providers=["IRIS", "http://example.com"])
        self.assertEqual(c._filter_requests(split), {})

    def test_downloading_waveforms(self):
        split = {
            "https://example.com": "1234",
            "http://example2.com": "1234",
            "http://example3.com": "1234",
            "http://service.iris.edu": "1234"
        }
        with mock.patch("obspy.clients.fdsn.client.Client") as p:
            mock_instance = p.return_value
            mock_instance.get_waveforms_bulk.return_value = obspy.read()
            # Only accept test1 as a an argument.
            mock_instance.services = {"dataselect": {"test1": True}}
            c = self._cls_object(debug=False, timeout=240)
            # test2 should not be passed on.
            st = c._download_waveforms(split=split, test1="a", test2="b")

        self.assertEqual(len(st), 12)
        # Test initialization.
        self.assertEqual(p.call_count, 4)
        self.assertEqual(set(_i[0][0] for _i in p.call_args_list),
                         set(split.keys()))
        self.assertEqual(set(_i[1]["debug"] for _i in p.call_args_list),
                         set([False]))
        self.assertEqual(set(_i[1]["timeout"] for _i in p.call_args_list),
                         set([240]))

        # Waveform download.
        wf_bulk = mock_instance.get_waveforms_bulk
        self.assertEqual(wf_bulk.call_count, 4)
        self.assertEqual(set(_i[0][0] for _i in wf_bulk.call_args_list),
                         set(["test1=a\n1234"]))
        for _i in wf_bulk.call_args_list:
            self.assertEqual(_i[1], {})

        # Once again, but raising exceptions this time.
        with mock.patch("obspy.clients.fdsn.client.Client") as p:
            mock_instance = p.return_value
            mock_instance.get_waveforms_bulk.side_effect = \
                FDSNNoDataException("No data")
            # Only accept test1 as a an argument.
            mock_instance.services = {"dataselect": {"test1": True}}
            c = self._cls_object(debug=False, timeout=240)
            # test2 should not be passed on.
            st = c._download_waveforms(split=split, test1="a", test2="b")

        self.assertEqual(len(st), 0)

        # Provider filtering might result in no data left.
        c.include_providers = "http://random.com"
        with self.assertRaises(FDSNNoDataException) as e:
            c._download_waveforms(split=split, test1="a", test2="b")
        self.assertEqual(
            e.exception.args[0],
            "Nothing remains to download after the provider "
            "inclusion/exclusion filters have been applied.")

    def test_downloading_stations(self):
        split = {
            "https://example.com": "1234",
            "http://example2.com": "1234",
            "http://example3.com": "1234",
            "http://service.iris.edu": "1234"
        }
        with mock.patch("obspy.clients.fdsn.client.Client") as p:
            mock_instance = p.return_value
            mock_instance.get_stations_bulk.return_value = \
                obspy.Inventory([], "")
            # Only accept test1 as a an argument.
            mock_instance.services = {"station": {"test1": True}}
            c = self._cls_object(debug=False, timeout=240)
            # test2 should not be passed on.
            c._download_stations(split=split, test1="a", test2="b")

        # Test initialization.
        self.assertEqual(p.call_count, 4)
        self.assertEqual(set(_i[0][0] for _i in p.call_args_list),
                         set(split.keys()))
        self.assertEqual(set(_i[1]["debug"] for _i in p.call_args_list),
                         set([False]))
        self.assertEqual(set(_i[1]["timeout"] for _i in p.call_args_list),
                         set([240]))

        # Station download.
        wf_bulk = mock_instance.get_stations_bulk
        self.assertEqual(wf_bulk.call_count, 4)
        self.assertEqual(set(_i[0][0] for _i in wf_bulk.call_args_list),
                         set(["test1=a\n1234"]))
        for _i in wf_bulk.call_args_list:
            self.assertEqual(_i[1], {})

    def test_unexpected_exception_handling(self):
        split = {
            "https://example.com": "1234"
        }

        with mock.patch("obspy.clients.fdsn.client.Client") as p:
            mock_instance = p.return_value
            mock_instance.get_stations_bulk.side_effect = ValueError("random")
            c = self._cls_object(debug=False, timeout=240)
            # Should not fail, but a warning should be raised.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                inv = c._download_stations(split=split)

        # Returns an empty inventory.
        self.assertIsInstance(inv, obspy.core.inventory.Inventory)
        self.assertEqual(len(inv), 0)

        # Raises a nice warning.
        self.assertEqual(len(w), 1)
        msg = w[0].message.args[0]
        self.assertTrue(msg.startswith(
            "Failed to download data of type 'station' from "
            "'https://example.com' due to:"))
        self.assertIn("ValueError: random", msg)
