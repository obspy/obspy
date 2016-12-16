#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
nrl.py for navigating, and picking responses from the
Nominal Response Library.

:copyright:
    Lloyd Carothers IRIS/PASSCAL, 2016
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import codecs
import io
import os
import sys
import warnings

import requests

if sys.version_info.major == 2:
    from urlparse import urljoin
    from ConfigParser import SafeConfigParser
else:
    from urllib.parse import urljoin
    from configparser import SafeConfigParser



class NRL:
    """
    Object representing the Nominal Response library.
    Can be Accessed online from the DMC or with a local copy
    Decision tree: Usually!!!! depth of tree and order can change!
    Sensor, make, model, period, sensitivity
    Datalogger, make, model, gain, sample rate

    The NRL has txt desision files formatted as windows ini

    """
    _index = 'index.txt'
    # Placholder for sample rate and gain
    SR = object()
    GAIN = object()
    dl_shortcuts = {
        'rt130': ['REF TEK', 'RT 130 & 130-SMA', GAIN, SR],
        'q330': ['Quanterra', 'Q330SR', GAIN, SR, 'LINEAR AT ALL SPS']}
    sensor_shortcuts = {
        'cmg3t': ['Guralp', 'CMG-3T', '120s - 50 Hz', '1500'],
        'trillium_240_1': [
            'Nanometrics', 'Trillium 240', '1 - serial numbers < 400'],
        'l22': ['Sercel/Mark Products', 'L-22D', '5470 Ohms', '20000 Ohms'],
        'sts2_g3': [
            'Streckeisen', 'STS-2', '1500', '3 - installed 04/97 to present']
        }

    def __init__(self, root='http://ds.iris.edu/NRL/'):
        if "://" in root:
            # use online, urls etc
            self._read_ini = self._read_ini_from_url
            self._read_resp = self._read_resp_from_url
            self._join = urljoin
            self._sep = '/'
            if not root.endswith('/'):
                root += '/'
            self._local = False
        else:
            # use local copy of NRL on filesystem
            self._read_ini = self._read_ini_from_filesystem
            self._read_resp = self._read_resp_from_filesystem
            self._join = self._join_filesystem
            self._sep = os.sep
            if not os.path.isdir(root):
                msg = "Not a local directory: '{}'".format(root)
                raise ValueError(msg)
            if not root.endswith(os.sep):
                root += os.sep
            self._local = True
        self.root = root
        self._sensors = None
        self._dataloggers = None

    def _join_filesystem(self, path1, path2):
        return os.path.join(os.path.dirname(path1), path2)

    def _read_ini_from_filesystem(self, path):
        # Don't use directly init sets read_ini()
        cp = SafeConfigParser()
        # XXX coding should be UTF-8 or ASCII??
        with codecs.open(path, mode='r', encoding='UTF-8') as f:
            if sys.version_info.major == 2:
                cp.readfp(f)
            else:
                cp.read_file(f)
        return cp

    def _read_ini_from_url(self, path):
        # Don't use directly init sets read_ini()
        cp = SafeConfigParser()
        response = requests.get(path)
        string_io = io.StringIO(response.text)
        if sys.version_info.major == 2:
            cp.readfp(string_io)
        else:
            cp.read_file(string_io)
        return cp

    def _read_resp_from_filesystem(self, path):
        # Returns Unicode string of RESP
        with open(path, 'r') as f:
            return f.read()

    def _read_resp_from_url(self, path):
        response = requests.get(path)
        return response.text

    def _print_ini(self, path):
        cp = self._read_ini(path)
        for section in cp.sections():
            print(section)
            for item in cp.items(section):
                print('\t', item)

    def choose(self, choice, path):
        # Should return either a path or a resp
        cp = self._read_ini(path)
        options = cp.options(choice)
        if 'path' in options:
            newpath = cp.get(choice, 'path')
        elif 'resp' in options:
            newpath = cp.get(choice, 'resp')
            # Strip quotes of new path
        if newpath.startswith('"'):
            newpath = newpath[1:]
        if newpath.endswith('"'):
            newpath = newpath[:-1]
        return self._join(path, newpath)

    def datalogger_path(self, answers, gain, sr):
        # Returns path of response file
        path = self.choose('Datalogger', self._join(self.root, self._index))
        # Fill placeholders for sr and gain
        answers_filled = list()
        for answer in answers:
            if answer is self.GAIN:
                answers_filled.append(str(gain))
            elif answer is self.SR:
                answers_filled.append(str(int(sr)))
            else:
                answers_filled.append(answer)
        for answer in answers_filled[:-1]:
            path = self.choose(str(answer), path)
        # path to RESP File
        return self._join(self.root, self.choose(str(answers_filled[-1]), path))

    def datalogger_path_from_short(self, shortname, gain, sr):
        return self.datalogger_path(NRL.dl_shortcuts[shortname], gain, sr)

    def datalogger_from_short(self, shortname, gain, sr):
        """
        Returns a unicode string of contents of RESP file
        """
        return self._read_resp(self.datalogger_path_from_short(
            shortname, gain, sr))

    def sensor_path(self, answers):
        # Returns path of response file
        # Args are usually: make, model, period, sensitivity
        path = self.choose('Sensor', self._join(self.root, self._index))
        # All but the last arg will return a ini file
        for answer in answers[:-1]:
            path = self.choose(str(answer), path)
        # The last arg returns a path to RESP
        return self._join(self.root, self.choose(str(answers[-1]), path))

    def sensor_path_from_short(self, shortname):
        return self.sensor_path(NRL.sensor_shortcuts[shortname])

    def sensor_from_short(self, shortname):
        """
        Returns a unicode string of contents of RESP file
        """
        return self._read_resp(self.sensor_path_from_short(shortname))

    def _recursive_parse(self, path):
        data = {}
        cp = self._read_ini(path)
        for section in cp.sections():
            if section == 'Main':
                continue
            for options in cp.options(section):
                if 'path' in options:
                    data[section] = self._recursive_parse(
                        self.choose(section, path))
                    # XXX is it safe to assume that if "path" is present we
                    # just follow down to the next level?
                    continue
                elif 'resp' in options:
                    data[section] = self._join(path, cp.get(section, 'resp'))
                    continue
                # XXX elif 'question' in options:
                # XXX     print(cp.get(section, 'question'))
        return data

    def parse_entire_library(self):
        """
        Parse the entire NRL
        """
        if not self._local:
            msg = ("Parsing the entire NRL library via http can take a *very* "
                   "long time.")
            warnings.warn(msg)
        self._sensors = self._recursive_parse(
            path=self._join(self.root, 'sensors' + self._sep + self._index))
        self._dataloggers = self._recursive_parse(
            path=self._join(self.root,
                            'dataloggers' + self._sep + self._index))


if __name__ == "__main__":
    import doctest
    doctest.testmod(exclude_empty=True)
