# -*- coding: utf-8 -*-
"""
The obspy.clients.syngine test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import unittest

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
        self.assertEqual(p.call_args[0][0],
                         'http://service.iris.edu/irisws/syngine/1/info')
        self.assertEqual(p.call_args[1]["params"],
                         {'model': 'test_model'})
        self.assertEqual(p.call_args[1]["headers"],
                         {'User-Agent': DEFAULT_TESTING_USER_AGENT})

    def test_get_available_models_mock(self):
        with mock.patch("requests.get") as p:
            p.return_value = RequestsMockResponse()
            self.c.get_available_models()

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[0][0],
                         'http://service.iris.edu/irisws/syngine/1/models')
        self.assertEqual(p.call_args[1]["params"], None)
        self.assertEqual(p.call_args[1]["headers"],
                         {'User-Agent': DEFAULT_TESTING_USER_AGENT})

    def test_print_model_information(self):
        with mock.patch("requests.get") as p:
            p.return_value = RequestsMockResponse()
            p.return_value._json = {"a": "b"}

            with CatchOutput() as out:
                self.c.print_model_information()

        self.assertEqual(out.stdout, b"{'a': 'b'}\n")

        self.assertEqual(p.call_count, 1)
        self.assertEqual(p.call_args[0][0],
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
        self.assertEqual(p.call_args[0][0],
                         'http://service.iris.edu/irisws/syngine/1/version')
        self.assertEqual(p.call_args[1]["params"], None)
        self.assertEqual(p.call_args[1]["headers"],
                         {'User-Agent': DEFAULT_TESTING_USER_AGENT})


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
