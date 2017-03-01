#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
obspy.clients.nrl.client.py for navigating, and picking responses from the
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
import requests
import sys

if sys.version_info.major == 2:
    import ConfigParser as configparser
else:
    import configparser


class NRL(object):
    """
    NRL client base class for accessing the Nominal Response Library:
    http://ds.iris.edu/NRL/

    Created with a URL for remote access or filesystem accessing a local copy.
    """
    _index = 'index.txt'
    def __new__(cls, root=None):
        try:
            o = requests.utils.urlparse(root)
        except AttributeError as e:
            # root is None
            o = None

        if root is None or o.scheme == 'http':
            # Create RemoteNRL
            return super(NRL, cls).__new__(RemoteNRL)
        elif os.path.isdir(o.path):
            # Create LocalNRL
            return super(NRL, cls).__new__(LocalNRL)
        else:
            raise TypeError('NRL requires a path or URL.')

    def __init__(self):
        self.sensors = self._parse_ini(self._join(self.root, 'sensors'))
        self.dataloggers = self._parse_ini(self._join(self.root, 'dataloggers'))

    def _choose(self, choice, path):
        # Should return either a path or a resp
        cp = self._get_cp_from_ini(path)
        options = cp.options(choice)
        if 'path' in options:
            newpath = cp.get(choice, 'path')
        elif 'resp' in options:
            newpath = cp.get(choice, 'resp')
        # Strip quotes of new path
        newpath = newpath.strip('"')
        # path = os.path.dirname(path)
        return self._join(path, newpath)

    def _parse_ini(self, path):
        if not path.endswith(self._index):
            path = self._join(path, self._index)
        print('parsing {}'.format(path))
        cp = self._get_cp_from_ini(path)

        nrl_dict = NRLDict(self)
        cp = self._get_cp_from_ini(path)
        for section in cp.sections():
            options = sorted(cp.options(section))
            if section.lower() == 'main':
                if options not in (['question'], ['detail', 'question']):
                    msg = "Unexpected structure of NRL file '{}'".format(path)
                    raise NotImplementedError(msg)
                nrl_dict._question = cp.get(section, 'question').strip('\'"')
                continue
            else:
                if options == ['path']:
                    nrl_dict[section] = NRLPath(self._choose(section, path))
                    continue
                # sometimes the description field is named 'description', but
                # sometimes also 'descr'
                elif options in (['description', 'resp'], ['descr', 'resp'],
                                 ['resp']):
                    if 'descr' in options:
                        descr = cp.get(section, 'descr')
                    elif 'description' in options:
                        descr = cp.get(section, 'description')
                    else:
                        descr = '<no description>'
                    nrl_dict[section] = (descr.strip('\'"'), self._join(
                        path, cp.get(section, 'resp').strip('\'"')))
                    continue
                else:
                    msg = "Unexpected structure of NRL file '{}'".format(path)
                    raise NotImplementedError(msg)
        return nrl_dict

    def get_parser(self, data):
        """
        Get  io.xseed.Parser for RESP
        """
        return 'xseed parser'

    def get_response(self, datalogger_keys, sensor_keys):
        """
        Get Response from NRL tree structure
    
        >>> nrl = NRL()  # doctest : +SKIP
        >>> response = nrl.get_response(  # doctest : +SKIP
        ...     sensor_keys=['Nanometrics', 'Trillium Compact', '120 s'],
        ...     datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])
        >>> print(response)  # doctest : +SKIP
        Channel Response
          From M/S () to COUNTS ()
          Overall Sensitivity: 4.74576e+08 defined at 1.000 Hz
          10 stages:
            Stage 1: PolesZerosResponseStage from M/S to V, gain: 754.3
            Stage 2: ResponseStage from V to V, gain: 1
            Stage 3: CoefficientsTypeResponseStage from V to COUNTS
            Stage 4: CoefficientsTypeResponseStage from COUNTS to COUNTS
            Stage 5: CoefficientsTypeResponseStage from COUNTS to COUNTS
            Stage 6: CoefficientsTypeResponseStage from COUNTS to COUNTS
            Stage 7: CoefficientsTypeResponseStage from COUNTS to COUNTS
            Stage 8: CoefficientsTypeResponseStage from COUNTS to COUNTS
            Stage 9: CoefficientsTypeResponseStage from COUNTS to COUNTS
            Stage 10: CoefficientsTypeResponseStage from COUNTS to COUNTS
    
        :type datalogger_keys: list of str
        :type sensor_keys: list of str
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        datalogger = self.dataloggers
        while datalogger_keys:
            datalogger = datalogger[datalogger_keys.pop(0)]
        datalogger_resp = self._read_resp(datalogger[1])
        dl_parser = Parser(datalogger_resp)
    
        sensor = self.sensors
        while sensor_keys:
            sensor = sensor[sensor_keys.pop(0)]
        sensor_resp = self._read_resp(sensor[1])
        sensor_parser = Parser(sensor_resp)
    
        resp_combined = Parser.combine_sensor_dl_resps(sensor_parser,
                                                           dl_parser)
        return Response.from_resp(resp_combined)

class NRLDict(dict):
    def __init__(self, nrl):
        self._nrl = nrl

    def __str__(self):
        if len(self):
            if self._question:
                info = ['{} ({} items):'.format(self._question, len(self))]
            else:
                info = ['{} items:'.format(len(self))]
            texts = ["'{}'".format(k) for k in sorted(self.keys())]
            info.extend(_textwrap(", ".join(texts), initial_indent='  ',
                                  subsequent_indent='  '))
            return '\n'.join(info)
        else:
            return '0 items.'

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __getitem__(self, name):
        value = super(NRLDict, self).__getitem__(name)
        # if encountering a not yet parsed NRL Path, expand it now
        if isinstance(value, NRLPath):
            value = self._nrl._parse_ini(value)
            self[name] = value
        return value


class NRLPath(str):
    pass


class LocalNRL(NRL):
    """
    Subclass of NRL for accessing local copy NRL.
    """
    def __init__(self, root):
        self.root = root
        self._join = os.path.join
        super(self.__class__, self).__init__()

    def _get_cp_from_ini(self, path):
        '''
        Returns a configparser from a path to an index.txt
        '''
        cp = configparser.SafeConfigParser()
        with codecs.open(path, mode='r', encoding='UTF-8') as f:
            if sys.version_info.major == 2:
                cp.readfp(f)
            else:
                cp.read_file(f)
        return cp

class RemoteNRL(NRL):
    """
    Subclass of NRL for accessing remote copy of NRL.
    """
    def __init__(self, root='http://ds.iris.edu/NRL/'):
        self.root = root
        super(self.__class__, self).__init__()

    def _join(self, *paths):
        url = paths[0]
        for path in paths[1:]:
            url = requests.compat.urljoin(url + '/', path)
        return url

    def _get_cp_from_ini(self, path):
        '''
        Returns a configparser from a path to an index.txt
        '''
        cp = configparser.SafeConfigParser()
        response = requests.get(path)
        string_io = io.StringIO(response.text)
        if sys.version_info.major == 2:
            cp.readfp(string_io)
        else:
            cp.read_file(string_io)
        return cp

if __name__ == "__main__":
    import doctest
    doctest.testmod(exclude_empty=True)
