#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest
import zipfile

from obspy.core.util import NamedTemporaryFile
from obspy.core.util.misc import CatchOutput, TemporaryWorkingDirectory
from obspy.io.xseed.parser import Parser
from obspy.io.xseed.scripts.dataless2resp import main as obspy_dataless2resp
from obspy.io.xseed.scripts.dataless2xseed import main as obspy_dataless2xseed
from obspy.io.xseed.scripts.xseed2dataless import main as obspy_xseed2dataless
from obspy.io.xseed.utils import compare_seed


class ScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.data = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data'))
        self.dataless_name = 'dataless.seed.BW_FURT'
        self.dataless_file = os.path.join(self.data, self.dataless_name)
        self.xseed_name = 'dataless.seed.BW_FURT.xml'
        self.xseed_file = os.path.join(self.data, self.xseed_name)

    #
    # obspy-dataless2resp
    #

    def test_dataless2resp(self):
        with TemporaryWorkingDirectory():
            with CatchOutput() as out:
                obspy_dataless2resp([self.dataless_file])

            expected = '''Found 1 files.
Parsing file %s
''' % (self.dataless_file,)
            self.assertEqual(expected, out.stdout)

            expected = ['RESP.BW.FURT..EHE',
                        'RESP.BW.FURT..EHN',
                        'RESP.BW.FURT..EHZ']
            actual = sorted(os.listdir(os.curdir))
            self.assertEqual(expected, actual)

    def test_dataless2resp_zipped(self):
        with TemporaryWorkingDirectory():
            with CatchOutput() as out:
                obspy_dataless2resp(['--zipped', self.dataless_file])

            expected = '''Found 1 files.
Parsing file %s
''' % (self.dataless_file,)
            self.assertEqual(expected, out.stdout)

            self.assertTrue(os.path.exists('dataless.seed.BW_FURT.zip'))

            expected = ['RESP.BW.FURT..EHE',
                        'RESP.BW.FURT..EHN',
                        'RESP.BW.FURT..EHZ']
            zf = zipfile.ZipFile('dataless.seed.BW_FURT.zip')
            actual = sorted(zf.namelist())
            zf.close()
            self.assertEqual(expected, actual)

    #
    # obspy-dataless2xseed
    #

    def test_dataless2xseed(self):
        with TemporaryWorkingDirectory():
            with CatchOutput() as out:
                obspy_dataless2xseed([self.dataless_file])

            expected = '''Found 1 files.
Parsing file %s
''' % (self.dataless_file,)
            self.assertEqual(expected, out.stdout)

            self.assertTrue(os.path.exists(self.xseed_name))

            with open(self.xseed_file, 'rt') as fh:
                expected = fh.read()
            with open(self.xseed_name, 'rt') as fh:
                actual = fh.read()

            self.assertEqual(expected, actual)

    def test_dataless2xseed_split(self):
        dataless_multi_file = os.path.join(self.data, 'CL.AIO.dataless')

        with TemporaryWorkingDirectory():
            with CatchOutput() as out:
                obspy_dataless2xseed(['--split-stations',
                                      dataless_multi_file])

            expected = '''Found 1 files.
Parsing file %s
''' % (dataless_multi_file,)
            self.assertEqual(expected, out.stdout)

            expected = ['CL.AIO.dataless.xml',
                        'CL.AIO.dataless.xml.1028697240.0.xml',
                        'CL.AIO.dataless.xml.1033117440.0.xml',
                        'CL.AIO.dataless.xml.1278350400.0.xml',
                        'CL.AIO.dataless.xml.1308244920.0.xml']
            actual = sorted(os.listdir(os.curdir))
            self.assertEqual(expected, actual)

    #
    # obspy-xseed2dataless
    #

    def test_xseed2dataless(self):
        with NamedTemporaryFile() as tf:
            with CatchOutput() as out:
                obspy_xseed2dataless(['--output', tf.name, self.xseed_file])

            expected = '''Found 1 files.
Parsing file %s
''' % (self.xseed_file,)
            self.assertEqual(expected, out.stdout)

            with open(self.dataless_file, 'rb') as fh:
                expected = fh.read()
            with open(tf.name, 'rb') as fh:
                actual = fh.read()

            try:
                compare_seed(expected, actual)
            except Exception:
                self.fail('compare_seed raised Exception unexpectedly!')
            parser1 = Parser(expected)
            parser2 = Parser(actual)
            self.assertEqual(parser1.get_seed(), parser2.get_seed())
