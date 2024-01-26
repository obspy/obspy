#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Client for accessing the `IRIS Library of Nominal Response for Seismic
Instruments <https://ds.iris.edu/NRL/>`_ (NRL).  To cite use of the NRL, please
see [Templeton2017]_.

:copyright:
    Lloyd Carothers IRIS/PASSCAL, 2016
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import codecs
import io
import os
import warnings
from configparser import ConfigParser
from urllib.parse import urlparse

import requests

import obspy
from obspy.core.inventory.util import _textwrap
from obspy.core.util.decorator import deprecated


# Simple cache for remote NRL access. The total data amount will always be
# fairly small so I don't think it needs any cache eviction for now.
_remote_nrl_cache = {}


class NRL(object):
    """
    NRL client base class for accessing the Nominal Response Library.

    https://ds.iris.edu/NRL/

    Created with a URL for remote access or filesystem accessing a local copy.

    .. warning::
        Remote access to online NRL is deprecated as it will stop working in
        Spring 2023 due to server side changes.
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
                       "but the directory does not exist.").format(root)
                raise ValueError(msg)
            return super(NRL, cls).__new__(LocalNRL)
        # Otherwise delegate to the remote NRL client to deal with all kinds
        # of remote resources (currently only HTTP).
        return super(NRL, cls).__new__(RemoteNRL)

    def __init__(self):
        try:
            sensor_index = self._join(self.root, 'sensors', self._index)
            self.sensors = self._parse_ini(sensor_index)

            datalogger_index = self._join(self.root, 'dataloggers',
                                          self._index)
            self.dataloggers = self._parse_ini(datalogger_index)
            # add empty dummies for integrated and soh which do not exist in
            # old NRL v1 just to be a little bit more consistent
            self.integrated = NRLDict(self)
            self.soh = NRLDict(self)
            self._nrl_version = 1
        except FileNotFoundError:
            sensor_index = self._join(self.root, 'sensor', self._index)
            self.sensors = self._parse_ini(sensor_index)

            datalogger_index = self._join(self.root, 'datalogger', self._index)
            self.dataloggers = self._parse_ini(datalogger_index)
            # version 2 also has additional base nodes "integrated" and "soh"
            integrated_index = self._join(self.root, 'integrated', self._index)
            self.integrated = self._parse_ini(integrated_index)
            soh_index = self._join(self.root, 'soh', self._index)
            self.soh = self._parse_ini(soh_index)
            self._nrl_version = 2

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
        elif 'xml' in options:
            newpath = cp.get(choice, 'xml')
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
                # NRL version 2 does not seem to have any of the 'descr = '
                # oddities anymore, but it can be downloaded in RESP format or
                # StationXML format and then the option name is different
                elif options in (['description', 'resp'], ['descr', 'resp'],
                                 ['resp'], ['description', 'xml']):
                    if 'descr' in options:
                        descr = cp.get(section, 'descr')
                    elif 'description' in options:
                        descr = cp.get(section, 'description')
                    else:
                        descr = '<no description>'
                    descr = self._clean_str(descr)
                    resp_path = self._choose(section, path)
                    if 'resp' in options:
                        resp_type = 'RESP'
                    elif 'xml' in options:
                        resp_type = 'STATIONXML'
                    else:
                        raise NotImplementedError(msg)
                    nrl_dict[section] = (descr, resp_path, resp_type)
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

        :type datalogger_keys: list[str]
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        response, _ = self._get_response('dataloggers', keys=datalogger_keys)
        first_stage = response.response_stages[0]

        if self._nrl_version == 2 and not first_stage.input_units:
            msg = ("Undefined input units in stage one. Most datalogger-only "
                   "responses in NRL v2 have a gain-only stage without units "
                   "specified as first stage. This has to be fixed manually "
                   "if necessary (first stage input units would usually be "
                   "'V' for volt). Also input units in overall instrument "
                   "sensitivity might have to be fixed manually.")
            warnings.warn(msg)
        if self._nrl_version == 2 \
                and first_stage.input_units.lower() in ("count", "counts"):
            msg = (f"First stage input units are '{first_stage.input_units}'. "
                   "When requesting a datalogger-only response from NRL v2, "
                   "for many cases the units of the first stage and also the "
                   "instrument overall sensitivity have to be fixed manually. "
                   "In general for these, input units should be 'V' for volt "
                   "and output units should be the same as second stage input "
                   "units, which usually are set correctly (either volt or "
                   "counts).")
            warnings.warn(msg)
        return response

    def get_sensor_response(self, sensor_keys):
        """
        Get the sensor response.

        :type sensor_keys: list[str]
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        response, _ = self._get_response('sensors', keys=sensor_keys)
        return response

    def get_integrated_response(self, keys):
        """
        Get an integrated response.

        :type keys: list[str]
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        if self._nrl_version == 1:
            msg = ('Integrated responses are only available in the new NRL v2 '
                   '(https://ds.iris.edu/ds/nrl/)')
            raise Exception(msg)
        response, _ = self._get_response('integrated', keys=keys)
        return response

    def get_soh_response(self, keys):
        """
        Get a SOH response.

        :type keys: list[str]
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        if self._nrl_version == 1:
            msg = ('SOH responses are only available in the new NRL v2 '
                   '(https://ds.iris.edu/ds/nrl/)')
            raise Exception(msg)
        response, _ = self._get_response('soh', keys=keys)
        return response

    def _get_response(self, base, keys):
        """
        Internal helper method to fetch a response

        This circumvents the warning message that is shown for NRL v2 when a
        datalogger-only response is fetched

        :type base: str
        :param base: either "sensors" or "dataloggers"
        :type keys: list of str
        :param keys: list of lookup keys
        """
        node = getattr(self, base)
        for key in keys:
            node = node[key]

        # Parse to an inventory object and return a response object.
        description, path, resp_type = node
        with io.BytesIO(self._read_resp(path).encode()) as buf:
            buf.seek(0, 0)
            return obspy.read_inventory(
                buf, format=resp_type)[0][0][0].response, resp_type

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
        ...     sensor_keys=['Nanometrics', 'Trillium Compact 120 (Vault, '
        ...                  'Posthole, OBS)', '754 V/m/s'],
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
        dl_resp, dl_resp_type = self._get_response(
            "dataloggers", keys=datalogger_keys)
        sensor_resp, sensor_resp_type = self._get_response(
            "sensors", keys=sensor_keys)
        return self._combine_sensor_datalogger(
            sensor_resp, dl_resp, sensor_resp_type, dl_resp_type)

    @staticmethod
    def _assert_units_ok(response):
        """
        Checks the units in the stage chain and overall sensitivity

        Raises an AssertionError if units are set but do not match throughout
        all stages and the instrument sensitivity. Raises other exceptions like
        IndexError if the assumptions that response stages are present and that
        there is an instrument sensitivity object are not met. Currently works
        case insensitive on unit name strings.

        :type response: :class:`~obspy.core.inventory.response.Response`
        """
        overall_input = response.instrument_sensitivity.input_units
        overall_output = response.instrument_sensitivity.output_units
        first_stage = response.response_stages[0]
        last_stage = response.response_stages[-1]
        if overall_input.lower() != first_stage.input_units.lower():
            msg = (f'Response has a unit mismatch (instrument sensitivity and '
                   f'first stage):\n{response}')
            raise AssertionError(msg)
        if overall_output.lower() != last_stage.output_units.lower():
            msg = (f'Response has a unit mismatch (instrument sensitivity and '
                   f'last stage):\n{response}')
            raise AssertionError(msg)
        for stage1, stage2 in zip(
                response.response_stages, response.response_stages[1:]):
            if stage1.output_units.lower() != stage2.input_units.lower():
                msg = (f'Response has a unit mismatch in the response chain:\n'
                       f'{response}')
                raise AssertionError(msg)

    def _combine_sensor_datalogger(
            self, sensor, datalogger, sensor_resp_type, datalogger_resp_type):
        """
        :type sensor: :class:`~obspy.core.inventory.response.Response`
        :type datalogger: :class:`~obspy.core.inventory.response.Response`
        :type sensor_resp_type: str
        :param sensor_resp_type: file format the sensor response was read from
        :type datalogger_resp_type: str
        :param datalogger_resp_type: file format the datalogger response was
            read from
        :rtype: :class:`~obspy.core.inventory.response.Response`
        """
        sensor_resp = sensor
        dl_resp = datalogger

        dl_first_stage = dl_resp.response_stages[0]
        dl_last_stage = dl_resp.response_stages[-1]
        try:
            sensor_stage0 = sensor_resp.response_stages[0]
        except IndexError:
            msg = ('Sensor response without stages (maybe polynomial only?) '
                   'not yet implemented. Please contact the developers.')
            raise NotImplementedError(msg)
        sensor_last_stage = sensor_resp.response_stages[-1]

        # information on changes between NRL v1 and v2:
        # https://ds.iris.edu/files/nrl/NominalResponseLibraryVersions.pdf
        if self._nrl_version == 1:
            # Combine both by replace stage one in the data logger with stage
            # one of the sensor.
            dl_resp.response_stages.pop(0)
            dl_resp.response_stages.insert(0, sensor_stage0)
        elif self._nrl_version == 2:
            # in principal we would simply want to avoid trying to fix units of
            # gain-only stages at this point in the get_datalogger_response()
            # call, because it would fix it half way and then trying to fix it
            # later would get skipped with the current implementation of
            # Response._attempt_to_fix_units().  However we would have to pass
            # an option through around seven function calls during reading
            # StationXML to get it down to io.stationxml.core._read_response()
            # so it would mean cluttering that module some, so for now it seems
            # less invasive to correct this problem later on, see below long
            # comment

            # check if stage numbering is sane in sensor response
            try:
                for i, stage in enumerate(sensor_resp.response_stages):
                    assert stage.stage_sequence_number == i + 1
            except AssertionError:
                msg = (f'Unexpected stage sequence numbering in sensor '
                       f'response:\n{str(sensor_resp)}')
                warnings.warn(msg)
            # check if stage numbering is sane in datalogger response
            try:
                for i, stage in enumerate(dl_resp.response_stages):
                    assert (
                        stage.stage_sequence_number ==
                        dl_resp.response_stages[0].stage_sequence_number + i)
            except AssertionError:
                msg = (f'Unexpected stage sequence numbering in datalogger '
                       f'response:\n{str(sensor_resp)}')
                warnings.warn(msg)

            # combine stages from sensor and datalogger
            for i, stage in enumerate(dl_resp.response_stages):
                stage.stage_sequence_number = \
                    len(sensor_resp.response_stages) + i + 1
            dl_resp.response_stages = (
                sensor_resp.response_stages + dl_resp.response_stages)
        else:
            raise NotImplementedError()

        dl_resp.instrument_sensitivity.input_units = sensor_stage0.input_units
        dl_resp.instrument_sensitivity.input_units_description = \
            sensor_stage0.input_units_description
        dl_resp.instrument_sensitivity.output_units = \
            dl_last_stage.output_units
        dl_resp.instrument_sensitivity.output_units_description = \
            dl_last_stage.output_units_description

        # NRLv2 seems to have two cases in terms of units that we need to take
        # care of..
        # - datalogger with a stage-gain-only minimal first stage
        #   without units and also its instrument sensitivity object lacking
        #   input units (which should be "V" for "volts")
        #   e.g. SeismicSource/Sigma4_PG1_FR250_DF0.1.xml
        #    -> should be fixed with `Response._attempt_to_fix_units()` after
        #       combining both sensor and datalogger
        # - datalogger with a stage-gain-only minimal stage without units in
        #   the middle of its response chain
        #   e.g. SeismicSource/Sigma4_PG1_FR250_DF0.1.xml
        #    -> should be fixed already during reading the datalogger only
        #       response file into an Inventory object due to
        #       `Response._attempt_to_fix_units()` getting called internally
        # - datalogger with other type of response stage (e.g. Poles and Zeros
        #   stage) as first stage
        #   e.g. WorldSensing/SpiderNano_PG8_FV5Vpp_FR500_FPMinimum.xml
        #    -> this is not covered in `Response._attempt_to_fix_units()` which
        #       only works on stage-gain-only stages, so setting the units of
        #       this kind of first datalogger stage to None and calling that
        #       method does not work. therefore check first if units are OK and
        #       only set them to None if really needed and if it is a
        #       stage-gain-only stage
        #  - datalogger with first stage as a stage-gain-only stage with tags
        #    for units but at least one of the units with an empty tag
        #    "<InputUnits><Name/></InputUnits>" which we currently parse into a
        #    value of '' (empty string) which the '_attempt_to_fix_units()'
        #    helper does not act upon.
        #    e.g. SolGeo/EDAX24_PG10_FR250.xml
        #     -> need to set units with value of empty string to None before
        #        calling the helper routine
        # Unfortunately units are halfway fixed during the read operation for
        # the datalogger-only response part (at least in the NRLv2 StationXML
        # variant), so calling that helper again does not do anything, unless
        # we set both input and output for that stage to None again.
        # In principle the cleaner solution would be to avoid calling
        # `Response._attempt_to_fix_units()` at the end of the read operation
        # on the datalogger-only response part, but that would mean adding a
        # lot of clutter by having to pass that option through a long chain of
        # function calls in io.stationxml.core only because of this edge case
        # of reading NRL responses, so the following seems overall cleaner and
        # should be safe anyway assuming the surrounding stages have valid
        # information on units, which they seem to have in NRLv2 database
        # In general this gain-only stage happens only as the first stage of
        # the datalogger-only response, but some few dataloggers also have a
        # gain-only stage without units in the middle of the response. But the
        # current approach is able to fill in the units and fix the unit chain
        # from the neighboring stages (e.g. the case for
        # 'datalogger/SeismicSource/Sigma4_PG1_FR250_DF0.1.xml')
        # Stages without units that are not stage-gain-only minimal stages are
        # currently not handled by `Response._attempt_to_fix_units()` and have
        # to be treated separately
        # Also see helper scripts in:
        #    https://github.com/megies/NRLv2-check-scripts
        if self._nrl_version == 2:
            # if reading data from RESP files it looks like the xseed Parser is
            # also trying to fix the initial stage units on the datalogger only
            # response.. set first stage input units to None and they should
            # get fixed later.
            # In principal we could just set datalogger first stage to "V" as
            # there are no other units in any sensor response but this should
            # be more general, just in case
            if datalogger_resp_type.upper() == 'RESP':
                dl_first_stage.input_units = None
                dl_first_stage.input_units_description = None
                dl_first_stage.output_units = None
                dl_first_stage.output_units_description = None
            # fix empty units that get parsed into an empty string instead of
            # None
            if dl_first_stage.input_units is not None and \
                    not dl_first_stage.input_units:
                dl_first_stage.input_units = sensor_last_stage.output_units
                dl_first_stage.input_units_description = \
                    sensor_last_stage.output_units_description
            dl_resp._attempt_to_fix_units()

        try:
            dl_resp.recalculate_overall_sensitivity()
        except ValueError:
            msg = "Failed to recalculate overall sensitivity."
            warnings.warn(msg)

        try:
            self._assert_units_ok(dl_resp)
        except AssertionError as e:
            warnings.warn(str(e))
        except Exception:
            msg = (f'Unexpected response (no stages or no instrument '
                   f'sensitivity?):\n{dl_resp}')
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
        cp = ConfigParser()
        with codecs.open(path, mode='r', encoding='UTF-8') as f:
            cp.read_file(f)
        return cp

    def _read_resp(self, path):
        # Returns Unicode string of RESP
        with open(path, 'r') as f:
            return f.read()


class RemoteNRL(NRL):
    """
    DEPRECATED

    Subclass of NRL for accessing remote copy of NRL.

    Direct access to online NRL is deprecated as it will stop working when the
    original NRLv1 gets taken offline (Spring 2023), please consider working
    locally with a downloaded full copy of the old NRLv1 or new NRLv2 following
    instructions on the
    `NRL landing page <https://ds.iris.edu/ds/nrl/>`_.
    """
    @deprecated()
    def __init__(self, root='https://ds.iris.edu/NRL'):
        """
        DEPRECATED

        Direct access to online NRL is deprecated as it will stop working when
        the original NRLv1 gets taken offline (Spring 2023), please consider
        working locally with a downloaded full copy of the old NRLv1 or new
        NRLv2 following instructions on the
        `NRL landing page <https://ds.iris.edu/ds/nrl/>`_.
        """
        self.root = root
        super(self.__class__, self).__init__()

    def _download(self, url):
        """
        Download service with basic cache.
        """
        if url not in _remote_nrl_cache:
            r = requests.get(url)
            _remote_nrl_cache[url] = r.text
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
        cp = ConfigParser()
        with io.StringIO(self._download(path)) as buf:
            cp.read_file(buf)
        return cp

    def _read_resp(self, path):
        return self._download(path)


if __name__ == "__main__":  # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
