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
    based on the network code but also more fine grained based on network and
    station code.
    The MultiClient can be extended by arbitrary user defined clients by adding
    to ``supported_client_types`` and ``supported_client_kwargs`` as long as
    the client has a ``get_waveforms()`` method with the same call syntax as
    obspy clients.

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
        # group lookups for network only and network+station
        self._lookup_net = {}
        self._lookup_netsta = {}
        # go through all SEED ID lookup keys
        for lookup_key, client_key in config.items('lookup'):
            level = lookup_key.count('.')
            if level == 0:
                self._lookup_net[lookup_key] = client_key
            elif level == 1:
                self._lookup_netsta[lookup_key] = client_key
            else:
                msg = (f"Invalid lookup key '{lookup_key}', should contain "
                       f"at most one dot ('.').")
                raise ValueError(msg)

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

    def _lookup_client_key(self, network, station):
        """
        Lookup and return the client key that should be used to fetch the data

        Raise an Exception if no client is defined via the config for this SEED
        ID.
        """
        if any(wildcard in network + station for wildcard in '?*'):
            msg = 'No wildcard characters allowed in network or station field'
            raise ValueError(msg)
        msg_multiple_matches = (
            f"Requested data '{network}.{station}' matches multiple lookup "
            f"keys of the same level defined in config ('{{}}'), using first "
            f"matching lookup key '{{}}' resolving to client key '{{}}'")
        # try to match starting on network+station level and then if no match
        # is found move to just network
        for lookup, seed_id in zip((self._lookup_netsta, self._lookup_net),
                                   (f'{network}.{station}', network)):
            # check for an exact match
            if seed_id in lookup:
                return lookup[seed_id]
            # if no direct match, go through and check also allowing for
            # wildcards
            matches = []
            for lookup_key in lookup:
                if fnmatch.fnmatch(seed_id, lookup_key):
                    matches.append(lookup_key)
            # no match found on this level, go to more general level
            if not matches:
                continue
            client_key = lookup[matches[0]]
            if len(matches) > 1:
                msg = msg_multiple_matches.format(
                    "', '".join(matches), matches[0], client_key)
                warnings.warn(msg)
            return client_key
        msg = (f"Found no matching lookup keys for requested data "
               f"'{network}.{station}'")
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
            request supports them). Unix shell-style wildcards (``[]``) are
            possible in principle, but most underlying client types might not
            support them in their ``get_waveforms()`` methods.
        :type channel: str
        :param channel: Channel code of requested data. Wildcards '?' and '*'
            are allowed (if the specific client that is eventually used for the
            request supports them). Unix shell-style wildcards (``[]``) are
            possible in principle, but most underlying client types might not
            support them in their ``get_waveforms()`` methods.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start time of requested waveforms data.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End time of requested waveforms data.
        :rtype: :class:`~obspy.core.stream.Stream`
        """
        client_key = self._lookup_client_key(network, station)
        client = self._get_or_initialize_client(client_key)
        return client.get_waveforms(network, station, location, channel,
                                    starttime, endtime)
