#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Client for accessing the Nominal Response Library (http://ds.iris.edu/NRL/).

:copyright:
    Lloyd Carothers IRIS/PASSCAL, 2016
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import codecs
import io
import os
import sys
import warnings

import requests

import obspy
from obspy.core.compatibility import (
    configparser, get_text_from_response, urlparse)
from obspy.core.inventory.util import _textwrap


# Simple cache for remote NRL access. The total data amount will always be
# fairly small so I don't think it needs any cache eviction for now.
_remote_nrl_cache = {}


class NRL(object):
    """
    NRL client base class for accessing the Nominal Response Library.

    http://ds.iris.edu/NRL/

    Created with a URL for remote access or filesystem accessing a local copy.
    """
    _index = 'index.txt'

    def __new__(cls, root=None):
        # root provided and it's no web URL
        if root:
            scheme = urlparse(root).scheme
            if scheme in ('http', 'https'):
                return super(NRL, cls).__new__(RemoteNRL)
            # Check if it's really a folder on the file-system.
            if not os.path.isdir(root):
                msg = ("Provided path '{}' seems to be a local file path "
                       "but the directory does not exist.").format(
                            root)
                raise ValueError(msg)
            return super(NRL, cls).__new__(LocalNRL)
        # Otherwise delegate to the remote NRL client to deal with all kinds
        # of remote resources (currently only HTTP).
        return super(NRL, cls).__new__(RemoteNRL)

    def __init__(self):
        sensor_index = self._join(self.root, 'sensors', self._index)
        self.sensors = self._parse_ini(sensor_index)

        datalogger_index = self._join(self.root, 'dataloggers', self._index)
        self.dataloggers = self._parse_ini(datalogger_index)

    def __str__(self):
        info = ['NRL library at ' + self.root]
        if self.sensors is None:
            info.append('  Sensors not parsed yet.')
        else:
            info.append(
                '  Sensors: {} manufacturers'.format(len(self.sensors)))
            if len(self.sensors):
                keys = [key for key in sorted(self.sensors)]
                lines = _textwrap("'" + "', '".join(keys) + "'",
                                  initial_indent='    ',
                                  subsequent_indent='    ')
                info.extend(lines)
        if self.dataloggers is None:
            info.append('  Dataloggers not parsed yet.')
        else:
            info.append('  Dataloggers: {} manufacturers'.format(
                len(self.dataloggers)))
            if len(self.dataloggers):
                keys = [key for key in sorted(self.dataloggers)]
                lines = _textwrap("'" + "', '".join(keys) + "'",
                                  initial_indent='    ',
                                  subsequent_indent='    ')
                info.extend(lines)
        return '\n'.join(_i.rstrip() for _i in info)

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(str(self))

    def _choose(self, choice, path):
        # Should return either a path or a resp
        cp = self._get_cp_from_ini(path)
        options = cp.options(choice)
        if 'path' in options:
            newpath = cp.get(choice, 'path')
        elif 'resp' in options:
            newpath = cp.get(choice, 'resp')
        # Strip quotes of new path
        newpath = self._clean_str(newpath)
        path = os.path.dirname(path)
        return self._join(path, newpath)

    def _parse_ini(self, path):
        nrl_dict = NRLDict(self)
        cp = self._get_cp_from_ini(path)
        for section in cp.sections():
            options = sorted(cp.options(section))
            if section.lower() == 'main':
                if options not in (['question'],
                                   ['detail', 'question']):  # pragma: no cover
                    msg = "Unexpected structure of NRL file '{}'".format(path)
                    raise NotImplementedError(msg)
                nrl_dict._question = self._clean_str(cp.get(section,
                                                            'question'))
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
                    descr = self._clean_str(descr)
                    resp_path = self._choose(section, path)
                    nrl_dict[section] = (descr, resp_path)
                    continue
                else:  # pragma: no cover
                    msg = "Unexpected structure of NRL file '{}'".format(path)
                    raise NotImplementedError(msg)
        return nrl_dict

    def _clean_str(self, string):
        return string.strip('\'"')

    def get_datalogger_response(self, datalogger_keys):
        """
        Get the datalogger response.

        :type datalogger_keys: list of str
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        datalogger = self.dataloggers
        for key in datalogger_keys:
            datalogger = datalogger[key]

        # Parse to an inventory object and return a response object.
        with io.BytesIO(self._read_resp(datalogger[1]).encode()) as buf:
            buf.seek(0, 0)
            return obspy.read_inventory(buf, format="RESP")[0][0][0].response

    def get_sensor_response(self, sensor_keys):
        """
        Get the sensor response.

        :type sensor_keys: list of str
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        sensor = self.sensors
        for key in sensor_keys:
            sensor = sensor[key]

        # Parse to an inventory object and return a response object.
        with io.BytesIO(self._read_resp(sensor[1]).encode()) as buf:
            buf.seek(0, 0)
            return obspy.read_inventory(buf, format="RESP")[0][0][0].response

    def get_response(self, datalogger_keys, sensor_keys):
        """
        Get Response from NRL tree structure

        :param datalogger_keys: List of data-loggers.
        :type datalogger_keys: list[str]
        :param sensor_keys: List of sensors.
        :type sensor_keys: list[str]
        :rtype: :class:`~obspy.core.inventory.response.Response`

        >>> nrl = NRL()
        >>> response = nrl.get_response(
        ...     sensor_keys=['Nanometrics', 'Trillium Compact', '120 s'],
        ...     datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])
        >>> print(response)   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Channel Response
          From M/S (Velocity in Meters per Second) to COUNTS (Digital Counts)
          Overall Sensitivity: 4.74576e+08 defined at 1.000 Hz
          10 stages:
            Stage 1: PolesZerosResponseStage from M/S to V, gain: 754.3
            Stage 2: ResponseStage from V to V, gain: 1
            Stage 3: Coefficients... from V to COUNTS, gain: 629129
            Stage 4: Coefficients... from COUNTS to COUNTS, gain: 1
            Stage 5: Coefficients... from COUNTS to COUNTS, gain: 1
            Stage 6: Coefficients... from COUNTS to COUNTS, gain: 1
            Stage 7: Coefficients... from COUNTS to COUNTS, gain: 1
            Stage 8: Coefficients... from COUNTS to COUNTS, gain: 1
            Stage 9: Coefficients... from COUNTS to COUNTS, gain: 1
            Stage 10: Coefficients... from COUNTS to COUNTS, gain: 1
        """
        dl_resp = self.get_datalogger_response(datalogger_keys)
        sensor_resp = self.get_sensor_response(sensor_keys)

        # Combine both by replace stage one in the data logger with stage
        # one of the sensor.
        dl_resp.response_stages.pop(0)
        dl_resp.response_stages.insert(0, sensor_resp.response_stages[0])
        try:
            dl_resp.recalculate_overall_sensitivity()
        except ValueError:
            msg = "Failed to recalculate overall sensitivity."
            warnings.warn(msg)

        return dl_resp


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
            return '\n'.join(_i.rstrip() for _i in info)
        else:
            return '0 items.'

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
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
        """
        Returns a configparser from a path to an index.txt
        """
        cp = configparser.ConfigParser()
        with codecs.open(path, mode='r', encoding='UTF-8') as f:
            if sys.version_info.major == 2:  # pragma: no cover
                cp.readfp(f)
            else:  # pragma: no cover
                cp.read_file(f)
        return cp

    def _read_resp(self, path):
        # Returns Unicode string of RESP
        with open(path, 'r') as f:
            return f.read()


class RemoteNRL(NRL):
    """
    Subclass of NRL for accessing remote copy of NRL.
    """
    def __init__(self, root='http://ds.iris.edu/NRL'):
        self.root = root
        super(self.__class__, self).__init__()

    def _download(self, url):
        """
        Download service with basic cache.
        """
        if url not in _remote_nrl_cache:
            r = requests.get(url)
            _remote_nrl_cache[url] = get_text_from_response(r)
        return _remote_nrl_cache[url]

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
        with io.StringIO(self._download(path)) as buf:
            if sys.version_info.major == 2:  # pragma: no cover
                cp.readfp(buf)
            else:  # pragma: no cover
                cp.read_file(buf)
        return cp

    def _read_resp(self, path):
        return self._download(path)


if __name__ == "__main__":  # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
