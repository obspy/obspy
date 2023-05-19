# -*- coding: utf-8 -*-
"""
Client for automatically fetching data through multiple individual clients from
submodules based on the requested SEED IDs and a given configuration.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import fnmatch
import warnings
from configparser import SafeConfigParser, NoOptionError, NoSectionError
from pathlib import Path

from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.clients.seedlink import Client as SeedlinkClient


class MultiClient(object):
    """
    Client for fetching waveform data from multiple sources

    MultiClient can be used to fetch waveform data in automated workflows from
    different clients from various submodules based on the SEED ID of the
    requested waveforms and a configuration that the user has to set up
    beforehand. In configuration the user can define which SEED ID should be
    fetched using what client. This can happend in the simplemost case just
    based on the network code but also more fine grained controlled by the full
    SEED ID and on in between levels of the SEED ID (station code etc.).
    The MultiClient can be extended by arbitrary user defined clients as well
    by adding to ``supported_client_types`` and ``supported_client_kwargs`` as
    long as the client has a ``get_waveforms()`` method with the same call
    syntax as obspy clients.

    :param config: Path to a local file with the textual config or file-like
        object containing the config.
    :type config: :class:`~pathlib.Path`, str or file-like object
    :param debug: Can be set to ``True`` to pass down the debug flag to all
        used clients. This might lead to a lot of printed debug output. The
        debug flag can also be set just for certain individual clients through
        the configuration like any other client parameter.
    """
    # supported clients, can be extended by the user if needed as long as the
    # client class has a get_waveforms() method with the usual call syntax
    supported_client_types = {
        "fdsn": FDSNClient,
        "seedlink": SeedlinkClient,
        "sds": SDSClient,
        }
    # explicitely specify what client kwargs we will look for in the config as
    # we need the information what type (str, int, ..) to use anyway (what
    # config getter method, to be precise)
    supported_client_kwargs = {
        "fdsn": {
            "base_url": "get", "user": "get", "password": "get",
            "user_agent": "get", "debug": "getboolean", "timeout": "getfloat"},
        "seedlink": {
            "server": "get", "port": "getint", "timeout": "getfloat",
            "debug": "getboolean"},
        "sds": {
            "sds_root": "get", "sds_type": "get", "format": "get",
            "fileborder_seconds": "getfloat", "fileborder_samples": "getint"},
        }

    def __init__(self, config, debug=False):
        self.debug = debug
        self._clients = {}
        self._parse_config_lookup_table(config)

    def _parse_config_lookup_table(self, path):
        """
        Parse config file

        :type path: :class:`pathlib.Path` or str
        """
        config = SafeConfigParser(allow_no_value=True)
        self._config = config
        # make all config keys case sensitive
        config.optionxform = str
        config.read(str(path))
        # group lookup dictionary by lookup level, key for the grouping will
        # simply be the number of '.' found in the key
        self._lookup = {0: {}, 1: {}, 2: {}, 3: {}}
        # go through all SEED ID lookup keys
        for lookup_key, client_key in config.items('lookup'):
            level = lookup_key.count('.')
            self._lookup[level][lookup_key] = client_key

    def _get_or_initialize_client(self, client_key):
        """
        Lazy client initialization only when given client is actually needed
        """
        if client_key in self._clients:
            return self._clients[client_key]
        client = self._initialize_client(client_key)
        return client

    def _initialize_client(self, client_key):
        """
        Initialize given client and store it for reuse
        """
        config = self._config
        # check if the server type is recognized
        client_type = config.get(client_key, "type")
        if client_type not in self.supported_client_types:
            msg = (f"Unknown client type '{client_type}' in client definition "
                   f"section '{client_key}' in config file.")
            raise NotImplementedError(msg)
        # parse parameters for client initialization from config
        kwargs = {}
        for key, getter in self.supported_client_kwargs[client_type].items():
            try:
                kwargs[key] = getattr(config, getter)(client_key, key)
            except NoOptionError:
                continue
        # override debug flag if requested on MultiClient init and if supported
        # by client type
        if self.debug and 'debug' in self.supported_client_kwargs[client_type]:
            kwargs['debug'] = True
        # initialize client
        client = self.supported_client_types[client_type](**kwargs)
        self._clients[client_key] = client
        return client

    def _lookup_client_key(self, network, station, location, channel):
        """
        Lookup and return the client key that should be used to fetch the data

        Raise an Exception if no client is defined via the config for this SEED
        ID.
        """
        if any(wildcard in network + station for wildcard in '?*'):
            msg = 'No wildcard characters allowed in network or station field'
            raise ValueError(msg)
        if any(special_character in network + station + location + channel for
               special_character in '[]'):
            msg = "Characters '[' and ']' are not allowed in any field"
            raise ValueError(msg)
        lookup = self._lookup
        nslc = '.'.join((network, station, location, channel))
        nsl = '.'.join((network, station, location))
        ns = '.'.join((network, station))
        n = network
        msg_multiple_matches = (
            "Requested SEED ID '{}' matches multiple lookup keys of the same "
            "level defined in config ('{}'), using first matching lookup key "
            "'{}'")
        # try to match starting on full SEED level and succesively falling back
        # to network level
        for level, seed_id in zip((3, 2, 1, 0), (nslc, nsl, ns, n)):
            # check for an exact match
            if seed_id in lookup[level]:
                return lookup[level][seed_id]
            # if there are wildcards on location/channel level, skip to higher
            # level if no exact match is found
            if any(wildcard in network + station for wildcard in '?*'):
                continue
            # if no direct match, go through and check also allowing for
            # wildcards
            matches = []
            for lookup_key in self._lookup[level].keys():
                if fnmatch.fnmatch(seed_id, lookup_key):
                    matches.append(lookup_key)
            if len(matches) > 1:
                msg = msg_multiple_matches.format(
                    nslc, "', '".join(matches), matches[0])
                warnings.warn(msg)
            # no match found on this level, go to more general level
            if not matches:
                continue
            return lookup[level][matches[0]]
        msg = f"Found no matching lookup keys for requested SEED ID: '{nslc}'"
        raise Exception(msg)

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime):
        """
        Get waveforms for given SEED ID

        :type network: str
        :param network: Network code of requested data. Wildcards are not
            allowed.
        :type station: str
        :param station: Station code of requested data. Wildcards are not
            allowed.
        :type location: str
        :param location: Location code of requested data. Wildcards '?' and '*'
            are allowed (if the specific client that is eventually used for the
            request supports them). However, if wildcards are used the 
            requested (wildcarded) SEED ID has to either be matched in exactly
            the same way in the config file as a lookup key down to location or
            channel level (e.g. requesting 'Z3.A100A.*.H*' there has to be a
            lookup key `Z3.A100A.*.H* = ...` in the configuration) or the
            lookup has to be specified on a higher level (e.g. `Z3.A100A = ...`
            or `Z3 = ...`)
        :type channel: str
        :param channel: Channel code of requested data. Wildcards '?' and '*'
            are allowed (if the specific client that is eventually used for the
            request supports them). However, if wildcards are used the
            requested (wildcarded) SEED ID has to either be matched in exactly
            the same way in the config file as a lookup key down to location or
            channel level (e.g. requesting 'Z3.A100A.*.H*' there has to be a
            lookup key `Z3.A100A.*.H* = ...` in the configuration) or the
            lookup has to be specified on a higher level (e.g. `Z3.A100A = ...`
            or `Z3 = ...`)
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start time of requested waveforms data.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End time of requested waveforms data.
        :rtype: :class:`~obspy.core.stream.Stream`
        """
        client_key = self._lookup_client_key(
            network, station, location, channel)
        client = self._get_or_initialize_client(client_key)
        return client.get_waveforms(network, station, location, channel,
                                    starttime, endtime)
