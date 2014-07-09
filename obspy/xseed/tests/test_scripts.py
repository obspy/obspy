#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.xseed.scripts.dataless2resp import main as obspy_dataless2resp
from obspy.xseed.scripts.dataless2xseed import main as obspy_dataless2xseed
from obspy.xseed.scripts.xseed2dataless import main as obspy_xseed2dataless
from obspy.xseed.parser import Parser
from obspy.xseed.utils import compareSEED
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.misc import CatchOutput, TemporaryWorkingDirectory
import zipfile
import os
import unittest


class ScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.dataless_name = 'dataless.seed.BW_FURT'
        self.dataless_file = os.path.join(os.path.dirname(__file__),
                                          'data',
                                          self.dataless_name)
        self.xseed_name = 'dataless.seed.BW_FURT.xml'
        self.xseed_file = os.path.join(os.path.dirname(__file__),
                                       'data',
                                       self.xseed_name)

    def test_dataless2resp(self):
        with TemporaryWorkingDirectory():
            with CatchOutput():
                obspy_dataless2resp([self.dataless_file])

            expected = ['RESP.BW.FURT..EHE',
                        'RESP.BW.FURT..EHN',
                        'RESP.BW.FURT..EHZ']
            actual = sorted(os.listdir(os.curdir))
            self.assertEqual(expected, actual)

    def test_dataless2resp_zipped(self):
        with TemporaryWorkingDirectory():
            with CatchOutput():
                obspy_dataless2resp(['--zipped', self.dataless_file])

            self.assertTrue(os.path.exists('dataless.seed.BW_FURT.zip'))

            expected = ['RESP.BW.FURT..EHE',
                        'RESP.BW.FURT..EHN',
                        'RESP.BW.FURT..EHZ']
            with zipfile.ZipFile('dataless.seed.BW_FURT.zip') as zf:
                actual = sorted(zf.namelist())
            self.assertEqual(expected, actual)

    def test_dataless2xseed(self):
        with TemporaryWorkingDirectory():
            with CatchOutput():
                obspy_dataless2xseed([self.dataless_file])

            self.assertTrue(os.path.exists(self.xseed_name))

            with open(self.xseed_file, 'rt') as fh:
                expected = fh.read()
            with open(self.xseed_name, 'rt') as fh:
                actual = fh.read()

            self.assertEqual(expected, actual)

    def test_dataless2xseed_split(self):
        dataless_multi_file = os.path.join(os.path.dirname(__file__),
                                           'data',
                                           'CL.AIO.dataless')

        with TemporaryWorkingDirectory():
            with CatchOutput():
                obspy_dataless2xseed(['--split-stations',
                                      dataless_multi_file])

            expected = ['CL.AIO.dataless.xml',
                        'CL.AIO.dataless.xml.1028697240.0.xml',
                        'CL.AIO.dataless.xml.1033117440.0.xml',
                        'CL.AIO.dataless.xml.1278350400.0.xml',
                        'CL.AIO.dataless.xml.1308244920.0.xml']
            actual = sorted(os.listdir(os.curdir))
            self.assertEqual(expected, actual)

    def test_xseed2dataless(self):
        with NamedTemporaryFile() as tf:
            with CatchOutput():
                obspy_xseed2dataless(['--output', tf.name, self.xseed_file])

            with open(self.dataless_file, 'rb') as fh:
                expected = fh.read()
            with open(tf.name, 'rb') as fh:
                actual = fh.read()

            try:
                compareSEED(expected, actual)
            except Exception:
                self.fail('compareSEED raised Exception unexpectedly!')
            parser1 = Parser(expected)
            parser2 = Parser(actual)
            self.assertEqual(parser1.getSEED(), parser2.getSEED())


def suite():
    return unittest.makeSuite(ScriptTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
