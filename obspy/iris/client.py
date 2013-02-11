# -*- coding: utf-8 -*-
"""
IRIS Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime, read, Stream
from obspy.core.event import readEvents
from obspy.core.util import NamedTemporaryFile, BAND_CODE, _getVersionString, \
    loadtxt
from urllib2 import HTTPError
import StringIO
import json
import os
import platform
import urllib
import urllib2
import warnings


VERSION = _getVersionString("obspy.iris")
DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (VERSION,
                                                   platform.platform(),
                                                   platform.python_version())
DEFAULT_PHASES = ['p', 's', 'P', 'S', 'Pn', 'Sn', 'PcP', 'ScS', 'Pdiff',
                  'Sdiff', 'PKP', 'SKS', 'PKiKP', 'SKiKS', 'PKIKP', 'SKIKS']


class Client(object):
    """
    IRIS Web service request client.

    :type base_url: str, optional
    :param base_url: Base URL of the IRIS Web service (default
        is ``'http://www.iris.edu/ws'``).
    :type user: str, optional
    :param user: The user name used for authentication with the Web
        service (default an empty string).
    :type password: str, optional
    :param password: A password used for authentication with the Web
        service (default is an empty string).
    :type timeout: int, optional
    :param timeout: Seconds before a connection timeout is raised (default
        is ``10`` seconds). This works only for Python >= 2.6.x.
    :type debug: bool, optional
    :param debug: Enables verbose output (default is ``False``).
    :type user_agent: str, optional
    :param user_agent: Sets an client identification string which may be
        used on server side for statistical analysis (default contains the
        current module version and basic information about the used
        operation system, e.g.
        ``'ObsPy 0.4.7.dev-r2432 (Windows-7-6.1.7601-SP1, Python 2.7.1)'``.

    .. rubric:: Example

    >>> from obspy.iris import Client
    >>> from obspy import UTCDateTime
    >>> client = Client()
    >>> t = UTCDateTime("2010-02-27T06:30:00.000")
    >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 20)
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    IU.ANMO.00.BHZ | 2010-02-27T06:30:00.019538Z - ... | 20.0 Hz, 401 samples
    """
    def __init__(self, base_url="http://www.iris.edu/ws",
                 user="", password="", timeout=20, debug=False,
                 user_agent=DEFAULT_USER_AGENT):
        """
        Initializes the IRIS Web service client.

        See :mod:`obspy.iris` for all parameters.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.debug = debug
        self.user_agent = user_agent
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(auth_handler)
        # install globally
        urllib2.install_opener(opener)

    def _fetch(self, url, data=None, headers={}, param_list=[], **params):
        """
        Send a HTTP request via urllib2.

        :type url: str
        :param url: Complete URL of resource
        :type data: str
        :param data: Channel list as returned by `availability` Web service
        :type headers: dict, optional
        :param headers: Additional header information for request
        """
        headers['User-Agent'] = self.user_agent
        # replace special characters
        remoteaddr = self.base_url + url
        options = '&'.join(param_list)
        if params:
            if options:
                options += '&'
            options += urllib.urlencode(params)
        if options:
            remoteaddr = "%s?%s" % (remoteaddr, options)
        if self.debug:
            print('\nRequesting %s' % (remoteaddr))
        req = urllib2.Request(url=remoteaddr, data=data, headers=headers)
        response = urllib2.urlopen(req, timeout=self.timeout)
        doc = response.read()
        return doc

    def _toFileOrData(self, filename, data, binary=False):
        """
        Either writes data into a file if filename is given or returns it.
        """
        if filename is None:
            return data
        if binary:
            method = 'wb'
        else:
            method = 'wt'
        # filename is given, create fh, write to file and return nothing
        if isinstance(filename, basestring):
            fh = open(filename, method)
        elif isinstance(filename, file):
            fh = filename
        else:
            msg = "Parameter filename must be either string or file handler."
            raise TypeError(msg)
        fh.write(data)
        fh.close()

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime, quality='B'):
        """
        Retrieves waveform data from IRIS and returns an ObsPy Stream object.

        :type network: str
        :param network: Network code, e.g. ``'IU'`` or ``'I*'``. Network code
            may contain wild cards.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'`` or ``'A*'``. Station code
            may contain wild cards.
        :type location: str
        :param location: Location code, e.g. ``'00'`` or ``'*'``. Location code
            may contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'`` or ``'B*'``. Channel code
            may contain wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type quality: ``'D'``, ``'R'``, ``'Q'``, ``'M'`` or ``'B'``, optional
        :param quality: Mini-SEED data quality indicator. ``'M'`` and ``'B'``
            (default) are treated the same and indicate best available.
            If ``'M'`` or ``'B'`` are selected, the output data records will be
            stamped with a ``'M'``.
        :return: ObsPy :class:`~obspy.core.stream.Stream` object.

        .. rubric:: Examples

        (1) Requesting waveform of a single channel.

            >>> from obspy.iris import Client
            >>> from obspy import UTCDateTime
            >>> client = Client()
            >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
            >>> t2 = UTCDateTime("2010-02-27T07:00:00.000")
            >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t1, t2)
            >>> print(st)  # doctest: +ELLIPSIS
            1 Trace(s) in Stream:
            IU.ANMO.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 36001 samples

        (2) Requesting waveforms of multiple channels using wildcard
            characters.

            >>> t1 = UTCDateTime("2010-084T00:00:00")
            >>> t2 = UTCDateTime("2010-084T00:30:00")
            >>> st = client.getWaveform("TA", "A25A", "", "BH*", t1, t2)
            >>> print(st)  # doctest: +ELLIPSIS
            3 Trace(s) in Stream:
            TA.A25A..BHE | 2010-03-25T00:00:00... | 40.0 Hz, 72001 samples
            TA.A25A..BHN | 2010-03-25T00:00:00... | 40.0 Hz, 72001 samples
            TA.A25A..BHZ | 2010-03-25T00:00:00... | 40.0 Hz, 72001 samples
        """
        kwargs = {}
        kwargs['network'] = str(network)[0:2]
        kwargs['station'] = str(station)[0:5]
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)[0:3]
        # try to be intelligent in starttime/endtime extension for fetching
        # data
        try:
            t_extension = 2.0 / BAND_CODE[kwargs['channel'][0]]
        except:
            # use 1 second extension if no proper bandcode info
            t_extension = 1.0
        kwargs['starttime'] = UTCDateTime(starttime) - t_extension
        kwargs['endtime'] = UTCDateTime(endtime) + t_extension
        if str(quality).upper() in ['D', 'R', 'Q', 'M', 'B']:
            kwargs['quality'] = str(quality).upper()

        # single channel request, go via `dataselect` Web service
        if all([val.isalnum() for val in (kwargs['network'],
                                          kwargs['station'],
                                          kwargs['location'],
                                          kwargs['channel'])]):
            st = self.dataselect(**kwargs)
        # wildcarded channel request, go via `availability` and
        # `bulkdataselect` Web services
        else:
            quality = kwargs.pop("quality", "")
            bulk = self.availability(**kwargs)
            st = self.bulkdataselect(bulk, quality)
        st.trim(UTCDateTime(starttime), UTCDateTime(endtime))
        return st

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, quality='B'):
        """
        Writes a retrieved waveform directly into a file.

        This method ensures the storage of the unmodified waveform data
        delivered by the IRIS server, e.g. preserving the record based
        quality flags of Mini-SEED files which would be neglected reading it
        with obspy.mseed.

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'IU'`` or ``'I*'``. Network code
            may contain wild cards.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'`` or ``'A*'``. Station code
            may contain wild cards.
        :type location: str
        :param location: Location code, e.g. ``'00'`` or ``'*'``. Location code
            may contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'`` or ``'B*'``. Channel code
            may contain wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type quality: ``'D'``, ``'R'``, ``'Q'``, ``'M'`` or ``'B'``, optional
        :param quality: Mini-SEED data quality indicator. ``'M'`` and ``'B'``
            (default) are treated the same and indicate best available.
            If ``'M'`` or ``'B'`` are selected, the output data records will be
            stamped with a ``'M'``.
        :return: ObsPy :class:`~obspy.core.stream.Stream` object.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime('2010-02-27T06:30:00.000')
        >>> t2 = UTCDateTime('2010-02-27T10:30:00.000')
        >>> client.saveWaveform('IU.ANMO.00.BHZ.mseed', 'IU', 'ANMO',
        ...                     '00', 'BHZ', t1, t2) # doctest: +SKIP
        """
        kwargs = {}
        kwargs['network'] = str(network)[0:2]
        kwargs['station'] = str(station)[0:5]
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)[0:3]
        kwargs['filename'] = str(filename)
        # try to be intelligent in starttime/endtime extension for fetching
        # data
        try:
            t_extension = 2.0 / BAND_CODE[kwargs['channel'][0]]
        except:
            # use 1 second extension if no proper bandcode info
            t_extension = 1.0
        kwargs['starttime'] = UTCDateTime(starttime) - t_extension
        kwargs['endtime'] = UTCDateTime(endtime) + t_extension
        if str(quality).upper() in ['D', 'R', 'Q', 'M', 'B']:
            kwargs['quality'] = str(quality).upper()
        self.dataselect(**kwargs)

    def saveResponse(self, filename, network, station, location, channel,
                     starttime, endtime, format='RESP'):
        """
        Writes response information into a file.

        Possible output formats are
        ``RESP`` (http://www.iris.edu/KB/questions/69/What+is+a+RESP+file%3F),
        ``StationXML`` (http://www.data.scec.org/xml/station/) or ``SACPZ``

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'IU'``.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``, wildcards allowed.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``, wildcards allowed.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: ``'RESP'``, ``'StationXML'`` or ``'SACPZ'``, optional
        :param format: Output format. Defaults to ``'RESP'``.
        """
        kwargs = {}
        kwargs['network'] = str(network)[0:2]
        kwargs['station'] = str(station)[0:5]
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)[0:3]
        kwargs['starttime'] = UTCDateTime(starttime)
        kwargs['endtime'] = UTCDateTime(endtime)
        # check format
        format = format.upper()
        if format == 'STATIONXML':
            # StationXML
            data = self.station(level='resp', **kwargs)
        elif format == 'SACPZ':
            # StationXML
            data = self.sacpz(**kwargs)
        elif format == 'RESP':
            # RESP
            data = self.resp(**kwargs)
        else:
            raise ValueError("Unsupported format %s" % format)
        fh = open(filename, "wb")
        fh.write(data)
        fh.close()

    def getEvents(self, format='catalog', **kwargs):
        """
        Retrieves event data from IRIS.

        :type format: ``'xml'`` or ``'catalog'``, optional
        :param format: Format of returned results. Defaults to ``'catalog'``.
        :rtype: :class:`~obspy.core.event.Catalog` or str
        :return: This method returns either a ObsPy
            :class:`~obspy.core.event.Catalog` object or a
            `QuakeML <https://quake.ethz.ch/quakeml/>`_ string depending on
            the given ``format`` keyword.

        **Geographic constraints - bounding rectangle**

        The following four parameters work together to specify a boundary
        rectangle. All four parameters are optional, but they may not be mixed
        with the parameters used for searching within a defined radius.

        :type minlat: float, optional
        :param minlat: Specify the southern boundary. The minimum latitude must
            be between -90 and 90 degrees inclusive (and less than or equal to
            maxlat). If not specified, then this value defaults to ``-90``.
        :type maxlat: float, optional
        :param maxlat: Specify the northern boundary. The maximum latitude must
            be between -90 and 90 degrees inclusive and greater than or equal
            to minlat. If not specified, then this value defaults to ``90``.
        :type minlon: float, optional
        :param minlon: Specify the western boundary. The minimum longitude must
            be between -180 and 180 degrees inclusive. If not specified, then
            this value defaults to ``-180``. If minlon > maxlon, then the
            boundary will cross the -180/180 meridian
        :type maxlon: float, optional
        :param maxlon: Specify the eastern boundary. The minimum longitude must
            be between -180 and 180 degrees inclusive. If not specified, then
            this value defaults to +180. If maxlon < minlon, then the boundary
            will cross the -180/180 meridian

        **Geographic constraints - bounding radius**

        The following four parameters work together to specify a boundary using
        a radius around a coordinate. ``lat``, ``lon``, and ``maxradius`` are
        all required, and must be used together. These parameters are
        incompatible with the boundary-box parameters described above.

        :type lat: float, optional
        :param lat: Specify the central latitude point, in degrees. This value
            must be between -90 and 90 degrees. This MUST be used in
            conjunction with the lon and maxradius parameters.
        :type lon: float, optional
        :param lon: Specify the central longitude point, in degrees. This MUST
            be used in conjunction with the lat and maxradius parameters.
        :type maxradius: float, optional
        :param maxradius: Specify the maximum radius, in degrees. Only
            earthquakes within maxradius degrees of the lat/lon point will be
            retrieved. This MUST be used in conjunction with the lat and lon
            parameters.
        :type minradius: float, optional
        :param minradius: This optional parameter allows for the exclusion of
            events that are closer than minradius degrees from the specified
            lat/lon point. This MUST be used in conjunction with the lat, lon,
            and maxradius parameters and is subject to the same restrictions.
            If this parameter isn't specified, then it defaults to ``0.0``
            degrees.

        **Depth constraints**

        :type mindepth: float, optional
        :param mindepth: Specify minimum depth (kilometers), values increase
            positively with depth, e.g. ``-1``.
        :type maxdepth: float, optional
        :param maxdepth: Specify maximum depth (kilometers), values increase
            positively with depth, e.g. ``20``.

        **Temporal constraints**

        The following two parameters impose time constrants on the query.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Limit results to the events occurring after the
            specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Limit results to the events occurring before the
            specified end time.

        **Magnitude constraints**

        :type minmag: float, optional
        :param minmag: Specify a minimum magnitude.
        :type maxmag: float, optional
        :param maxmag: Specify a maximum magnitude.
        :type magtype: string, optional
        :param magtype: Specify magnitude type. Some common types (there are
            many) include ``"Ml"`` (local/Richter magnitude), ``"Ms"`` (surface
            magnitude), ``"mb"`` (body wave magnitude), ``"Mw"`` (moment
            magnitude).

        **Subset the data by organization**

        :type catalog: string, optional
        :param catalog: Specify a catalog
            [`available catalogs <http://www.iris.edu/ws/event/catalogs>`_].
            Results will include any origins which contain the specified
            catalog text, i.e. ``"PDE"`` will match ``"NEIC PDE"``
        :type contributor: string, optional
        :param contributor: Specify a contributor [`available
            contributors <http://www.iris.edu/ws/event/contributors>`_]. When
            selecting a contributor, the result includes a preferred origin as
            specified by the contributor. Results will include any origins
            which contain the specified contributor text, i.e. ``"NEIC"`` will
            match ``"NEIC PDE-Q"``.

        **Miscellaneous parameters**

        These parameters affect how the search is conducted, and how the events
        are returned.

        :type limit: int, optional
        :param limit: Limit the results to the specified number of events. This
            value must be 10 or greater. By default, the results are not
            limited.
        :type orderby: ``"time"`` or ``"magnitude"``, optional
        :param orderby: Sort the resulting events in order of descending time
            or magnitude. By default, results are sorted in descending time
            order.
        :type updatedafter:  :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param updatedafter: Select origins updated after a certain date. This
            is most useful for synchronization purposes.
        :type includeallmagnitudes: bool, optional
        :param includeallmagnitudes: This will include all the magnitudes in
            search and print criteria. If magnitudes do not exist for a certain
            origin, the search algorithm will consider it a miss and therefore
            will not include the event. Defaults to ``True``.
        :type includearrivals: bool, optional
        :param includearrivals: If this event has associated phase arrival
            information, then it will be included in the results. Defaults to
            ``False``.
        :type preferredonly: bool, optional
        :param preferredonly: Include preferred estimates only. When catalog is
            selected, the result returned will include the preferred origin as
            specified by the catalog. Defaults to ``True``.

        **Specifying an event using an IRIS ID number**

        Individual events can be retrieved using ID numbers assigned by IRIS.
        When these parameters are used, then only the ``includeallmagnitudes``
        and ``preferredonly`` parameters are also allowed.

        :type eventid: int, optional
        :param eventid: Retrieve an event based on the unique IRIS event id.
        :type originid: int, optional
        :param originid: Retrieve an event based on the unique IRIS origin id.
        :type magnitudeid: int, optional
        :param magnitudeid: Retrieve an event based on the unique IRIS
            magnitude id.

        The IRIS DMC receives earthquake location and magnitude information
        primarily from the
        `USGS NEIC <http://earthquake.usgs.gov/regional/neic/>`_ and the
        `ISC <http://www.isc.ac.uk/>`_, other sources include the
        `Global CMT project <http://www.globalcmt.org/>`_ and the
        `USArray ANF <http://anf.ucsd.edu/>`_.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> client = Client()
        >>> events = client.getEvents(format='xml', minmag=9.1)
        >>> print(events)  # doctest: +ELLIPSIS
        <q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" ...
        """
        # fetch data
        data = self.event(**kwargs)
        # format output
        if format == "catalog":
            return readEvents(StringIO.StringIO(data), 'QUAKEML')
        return data

    def timeseries(self, network, station, location, channel,
                   starttime, endtime, filter=[], filename=None,
                   output='miniseed', **kwargs):
        """
        Low-level interface for `timeseries` Web service of IRIS
        (http://www.iris.edu/ws/timeseries/)- release 1.3.5 (2012-06-07).

        This method fetches segments of seismic data and returns data formatted
        in either MiniSEED, ASCII or SAC. It can optionally filter the data.

        **Channel and temporal constraints (required)**

        The four SCNL parameters (Station - Channel - Network - Location) are
        used to determine the channel of interest, and are all required.
        Wildcards are not accepted.

        :type network: str
        :param network: Network code, e.g. ``'IU'``.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.

        **Filter Options**

        The following parameters act as filters upon the timeseries.

        :type filter: list of str, optional
        :param filter: Filter list.  List order matters because each filter
            operation is performed in the order given. For example
            ``filter=["demean", "lp=2.0"]`` will demean and then apply a
            low-pass filter, while ``filter=["lp=2.0", "demean"]`` will apply
            the low-pass filter first, and then demean.

            ``"taper=WIDTH,TYPE"``
                Apply a time domain symmetric tapering function to the
                timeseries data. The width is specified as a fraction of the
                trace length from 0 to 0.5. The window types HANNING (default),
                HAMMING, or COSINE may be optionally followed, e.g.
                ``"taper=0.25"`` or ``"taper=0.5,COSINE"``.
            ``"envelope=true"``
                Calculate the envelope of the time series. This calculation
                uses a Hilbert transform approximated by a time domain filter.
            ``"lp=FREQ"``
                Low-pass filter the time-series using an IIR 4th order filter,
                using this value (in Hertz) as the cutoff, e.g. ``"lp=1.0"``.
            ``"hp=FREQ"``
                High-pass filter the time-series using an IIR 4th order filter,
                using this value (in Hertz) as the cutoff, e.g. ``"hp=3.0"``.
            ``"bp=FREQ1,FREQ2"``
                Band pass frequencies, in Hz, e.g. ``"bp=0.1,1.0"``.
            ``"demean"``
                Remove mean value from data.
            ``"scale"``
                Scale data samples by specified factor, e.g. ``"scale=2.0"``
                If ``"scale=AUTO"``, the data will be scaled by the stage-zero
                gain. Cannot specify both ``scale`` and ``divscale``.
                Cannot specify both ``correct`` and ``scale=AUTO``.
            ``"divscale"``
                Scale data samples by the inverse of the specified factor, e.g
                ``"divscale=2.0"``. You cannot specify both ``scale`` and
                ``divscale``.
            ``"correct"``
                Apply instrument correction to convert to earth units. Uses
                either deconvolution or polynomial response correction. Cannot
                specify both ``correct`` and ``scale=AUTO``. Correction
                on > 10^7 samples will result in an error. At a sample rate of
                20 Hz, 10^7 samples is approximately 5.8 days.
            ``"freqlimits=FREQ1,FREQ2,FREQ3,FREQ4"``
                Specify an envelope for a spectrum taper for deconvolution,
                e.g. ``"freqlimits=0.0033,0.004,0.05,0.06"``. Frequencies are
                specified in Hertz. This cosine taper scales the spectrum from
                0 to 1 between FREQ1 and FREQ2 and from 1 to 0 between FREQ3
                and FREQ4. Can only be used with the correct option. Cannot be
                used in combination with the ``autolimits`` option.
            ``"autolimits=X,Y"``
                Automatically determine frequency limits for deconvolution,
                e.g. ``"autolimits=3.0,3.0"``. A pass band is determined for
                all frequencies with the lower and upper corner cutoffs defined
                in terms of dB down from the maximum amplitude. This algorithm
                is designed to work with flat responses, i.e. a response in
                velocity for an instrument which is flat to velocity. Other
                combinations will likely result in unsatisfactory results.
                Cannot be used in combination with the ``freqlimits`` option.
            ``"units=UNIT"``
                Specify output units. Can be DIS, VEL, ACC or DEF, where DEF
                results in no unit conversion, e.g. ``"units=VEL"``. Option
                ``units`` can only be used with ``correct``.
            ``"diff"``
                Differentiate using 2 point (uncentered) method
            ``"int"``
                Integrate using trapezoidal (midpoint) method
            ``"decimate=SAMPLERATE"``
                Specify the sample-rate to decimate to, e.g.
                ``"decimate=2.0"``. The sample-rate of the source divided by
                the given sample-rate must be factorable by 2,3,4,7.

        **Miscelleneous options**

        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :type output: str, optional
        :param output: Output format if parameter ``filename`` is used.

            ``'ascii'``
                Data format, 1 column (values)
            ``'ascii2'``
                ASCII data format, 2 columns (time, value)
            ``'ascii'``
                Same as ascii2
            ``'audio'``
                audio WAV file
            ``'miniseed'``
                IRIS miniSEED format
            ``'plot'``
                A simple plot of the timeseries
            ``'saca'``
                SAC, ASCII format
            ``'sacbb'``
                SAC, binary big-endian format
            ``'sacbl'``
                SAC, binary little-endian format

        :rtype: :class:`~obspy.core.stream.Stream` or ``None``
        :return: ObsPy Stream object if no ``filename`` is given.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> dt = UTCDateTime("2005-01-01T00:00:00")
        >>> client = Client()
        >>> st = client.timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
        >>> print(st[0].data)  # doctest: +ELLIPSIS
        [  24   20   19   19   19   15   10    4   -4  -11 ...
        >>> st = client.timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10,
        ...     filter=["correct", "demean", "lp=2.0"])
        >>> print(st[0].data)  # doctest: +ELLIPSIS
        [ -1.38267058e-06  -1.10900783e-06  -6.89020794e-07 ...
        """
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        # convert UTCDateTime to string for query
        kwargs['starttime'] = UTCDateTime(starttime).formatIRISWebService()
        kwargs['endtime'] = UTCDateTime(endtime).formatIRISWebService()
        # output
        if filename:
            kwargs['output'] = output
        else:
            kwargs['output'] = 'miniseed'
        # build up query
        url = '/timeseries/query'
        try:
            data = self._fetch(url, param_list=filter, **kwargs)
        except HTTPError, e:
            msg = "No waveform data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        # write directly if filename is given
        if filename:
            return self._toFileOrData(filename, data, True)
        # create temporary file for writing data
        tf = NamedTemporaryFile()
        tf.write(data)
        # read stream using obspy.mseed
        tf.seek(0)
        try:
            stream = read(tf.name, 'MSEED')
        except:
            stream = Stream()
        tf.close()
        # remove temporary file:
        try:
            os.remove(tf.name)
        except:
            pass
        return stream

    def resp(self, network, station, location="*", channel="*",
             starttime=None, endtime=None, filename=None, **kwargs):
        """
        Low-level interface for `resp` Web service of IRIS
        (http://www.iris.edu/ws/resp/) - 1.4.1 (2011-04-14).

        This method provides access to channel response information in the SEED
        `RESP <http://www.iris.edu/KB/questions/69/What+is+a+RESP+file%3F>`_
        format (as used by evalresp). Users can query for channel response by
        network, station, channel, location and time.

        :type network: str
        :param network: Network code, e.g. ``'IU'``.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'``.
        :type location: str, optional
        :param location: Location code, e.g. ``'00'``, wildcards allowed.
            Defaults to ``'*'``.
        :type channel: str, optional
        :param channel: Channel code, e.g. ``'BHZ'``, wildcards allowed.
            Defaults to ``'*'``.

        **Temporal constraints**

        The following three parameters impose time constrants on the query.
        Time may be requested through the use of either time OR the start and
        end times. If no time is specified, then the current time is assumed.

        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param time: Find the response for the given time. Time cannot be used
            with starttime or endtime parameters
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Start time, may be used in conjunction with endtime.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: End time, may be used in conjunction with starttime.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: str or ``None``
        :return: SEED RESP file as string if no ``filename`` is given..

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> dt = UTCDateTime("2010-02-27T06:30:00.000")
        >>> data = client.resp("IU", "ANMO", "00", "BHZ", dt)
        >>> print(data)  # doctest: +ELLIPSIS
        #
        ####################################################################...
        #
        B050F03     Station:     ANMO
        B050F16     Network:     IU
        B052F03     Location:    00
        B052F04     Channel:     BHZ
        ...
        """
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        # convert UTCDateTime to string for query
        if starttime and endtime:
            try:
                kwargs['starttime'] = \
                    UTCDateTime(starttime).formatIRISWebService()
            except:
                kwargs['starttime'] = starttime
            try:
                kwargs['endtime'] = UTCDateTime(endtime).formatIRISWebService()
            except:
                kwargs['endtime'] = endtime
        elif 'time' in kwargs:
            try:
                kwargs['time'] = \
                    UTCDateTime(kwargs['time']).formatIRISWebService()
            except:
                pass
        # build up query
        url = '/resp/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return self._toFileOrData(filename, data)

    def station(self, network, station, location="*", channel="*",
                starttime=None, endtime=None, level='sta', filename=None,
                **kwargs):
        """
        Low-level interface for `station` Web service of IRIS
        (http://www.iris.edu/ws/station/) - release 1.3.6 (2012-04-30).

        This method provides access to station metadata in the IRIS DMC
        database. The results are returned in XML format using the StationXML
        schema (http://www.data.scec.org/xml/station/). Users can query for
        station metadata by network, station, channel, location, time and other
        search criteria and request results at multiple levels (station,
        channel, response, etc.).

        :type network: str
        :param network: Network code, e.g. ``'IU'``.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'``, wildcards allowed.
        :type location: str, optional
        :param location: Location code, e.g. ``'00'``, wildcards allowed.
            Defaults to ``'*'``.
        :type channel: str, optional
        :param channel: Channel code, e.g. ``'BHZ'``, wildcards allowed.
            Defaults to ``'*'``.

        **Geographic constraints - bounding rectangle**

        The following four parameters work together to specify a boundary
        rectangle. All four parameters are optional, but they may not be mixed
        with the parameters used for searching within a defined radius.

        :type minlat: float, optional
        :param minlat: Specify the southern boundary. The minimum latitude must
            be between -90 and 90 degrees inclusive (and less than or equal to
            maxlat). If not specified, then this value defaults to ``-90``.
        :type maxlat: float, optional
        :param maxlat: Specify the northern boundary. The maximum latitude must
            be between -90 and 90 degrees inclusive and greater than or equal
            to minlat. If not specified, then this value defaults to ``90``.
        :type minlon: float, optional
        :param minlon: Specify the western boundary. The minimum longitude must
            be between -180 and 180 degrees inclusive. If not specified, then
            this value defaults to ``-180``. If minlon > maxlon, then the
            boundary will cross the -180/180 meridian
        :type maxlon: float, optional
        :param maxlon: Specify the eastern boundary. The minimum longitude must
            be between -180 and 180 degrees inclusive. If not specified, then
            this value defaults to +180. If maxlon < minlon, then the boundary
            will cross the -180/180 meridian

        **Geographic constraints - bounding radius**

        The following four parameters work together to specify a circular
        bounding area. ``lat``, ``lon``, and ``maxradius`` are all required,
        and must be used together. ``minradius`` is optional, and defaults
        to ``0``. These parameters are incompatible with the boundary-box
        parameters described above.

        :type lat: float, optional
        :param lat: Specify the central latitude point, in degrees. This value
            must be between -90 and 90 degrees. This MUST be used in
            conjunction with the lon and maxradius parameters.
        :type lon: float, optional
        :param lon: Specify the central longitude point, in degrees. This MUST
            be used in conjunction with the lat and maxradius parameters.
        :type maxradius: float, optional
        :param maxradius: Specify the maximum radius, in degrees. Only
            earthquakes within maxradius degrees of the lat/lon point will be
            retrieved. This MUST be used in conjunction with the lat and lon
            parameters.
        :type minradius: float, optional
        :param minradius: This optional parameter allows for the exclusion of
            events that are closer than minradius degrees from the specified
            lat/lon point. This MUST be used in conjunction with the lat, lon,
            and maxradius parameters and is subject to the same restrictions.
            If this parameter isn't specified, then it defaults to ``0.0``
            degrees.

        **Temporal constraints**

        The following parameters impose various time constrants on the query.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Limit results to the stations that were operational
            on or after this time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Limit results to the stations that were operational on
            or before this time.
        :type startbefore: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param startbefore: Limit results to the stations starting before this
            time.
        :type startafter: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param startafter: Limit results to the stations starting after this
            time.
        :type endbefore: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endbefore: Limit results to the stations ending before this
            time.
        :type endafter: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endafter: Limit results to the stations ending after this time.

        **Miscelleneous options**

        :type updatedafter: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param updatedafter: Only show stations that were updated after a
            specific time.
        :type level: ``'net'``, ``'sta'``, ``'chan'``, or ``'resp'``, optional
        :param level: Specify whether to include channel/response metadata or
            not. Defaults to ``'sta'``.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: str or ``None``
        :return: StationXML file as string if no ``filename`` is given.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2006-03-01")
        >>> t2 = UTCDateTime("2006-09-01")
        >>> station_xml = client.station(network="IU", station="ANMO",
        ...                              location="00", channel="BHZ",
        ...                              starttime=t1, endtime=t2, level="net")
        >>> print(station_xml) # doctest: +ELLIPSIS
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <BLANKLINE>
        <StaMessage ...>
         ...
         <Network net_code="IU">
          <StartDate>1988-01-01T00:00:00</StartDate>
          <EndDate>2500-12-12T23:59:59</EndDate>
          <Description>Global Seismograph Network ...</Description>
          <TotalNumberStations>91</TotalNumberStations>
          <SelectedNumberStations>0</SelectedNumberStations>
         </Network>
        </StaMessage>
        """
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        kwargs['level'] = level
        # convert UTCDateTimes to string
        if starttime and endtime:
            try:
                kwargs['starttime'] = \
                    UTCDateTime(starttime).formatIRISWebService()
            except:
                kwargs['starttime'] = starttime
        if endtime:
            try:
                kwargs['endtime'] = UTCDateTime(endtime).formatIRISWebService()
            except:
                kwargs['endtime'] = endtime
        for key in ['startbefore', 'startafter', 'endbefore', 'endafter',
                    'updatedafter']:
            try:
                kwargs[key] = \
                    UTCDateTime(kwargs[key]).formatIRISWebService()
            except KeyError:
                pass
        # build up query
        url = '/station/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return self._toFileOrData(filename, data)

    def dataselect(self, network, station, location, channel,
                   starttime, endtime, quality='B', filename=None, **kwargs):
        """
        Low-level interface for `dataselect` Web service of IRIS
        (http://www.iris.edu/ws/dataselect/)- release 1.8.1 (2012-05-03).

        This method returns a single channel of time series data (no wildcards
        are allowed). With this service you specify network, station, location,
        channel and a time range and the service returns either as an ObsPy
        :class:`~obspy.core.stream.Stream` object or saves the data directly
        as Mini-SEED file.

        :type network: str
        :param network: Network code, e.g. ``'IU'``.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type quality: ``'D'``, ``'R'``, ``'Q'``, ``'M'`` or ``'B'``, optional
        :param quality: Mini-SEED data quality indicator. ``'M'`` and ``'B'``
            (default) are treated the same and indicate best available.
            If ``'B'`` is selected, the output data records will be stamped
            with a ``M``.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: :class:`~obspy.core.stream.Stream` or ``None``
        :return: ObsPy Stream object if no ``filename`` is given.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2010-02-27T06:30:00.000")
        >>> t2 = UTCDateTime("2010-02-27T07:00:00.000")
        >>> st = client.dataselect("IU", "ANMO", "00", "BHZ", t1, t2)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 36000 samples
        """
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        kwargs['quality'] = str(quality)
        # convert UTCDateTime to string for query
        kwargs['starttime'] = UTCDateTime(starttime).formatIRISWebService()
        kwargs['endtime'] = UTCDateTime(endtime).formatIRISWebService()
        # build up query
        url = '/dataselect/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            msg = "No waveform data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        # write directly if filename is given
        if filename:
            return self._toFileOrData(filename, data, True)
        # create temporary file for writing data
        tf = NamedTemporaryFile()
        tf.write(data)
        # read stream using obspy.mseed
        tf.seek(0)
        try:
            stream = read(tf.name, 'MSEED')
        except:
            stream = Stream()
        tf.close()
        # remove temporary file:
        try:
            os.remove(tf.name)
        except:
            pass
        return stream

    def bulkdataselect(self, bulk, quality=None, filename=None,
                       minimumlength=None, longestonly=True):
        """
        Low-level interface for `bulkdataselect` Web service of IRIS
        (http://www.iris.edu/ws/bulkdataselect/) - release 1.4.5 (2012-05-03).

        This method returns multiple channels of time series data for specified
        time ranges. With this service you specify a list of selections
        composed of network, station, location, channel, starttime and endtime
        and the service streams back the selected raw waveform data as an ObsPy
        :class:`~obspy.core.stream.Stream` object.

        Simple requests with wildcards can be performed via
        :meth:`~obspy.iris.client.Client.getWaveform`. The list with channels
        can also be generated using
        :meth:`~obspy.iris.client.Client.availability`.

        :type bulk: str
        :param bulk: List of channels to fetch as returned by
            :meth:`~obspy.iris.client.Client.availability`.
            Can be a filename with a text file in bulkdataselect compatible
            format or a string in the same format.
        :type quality: ``'D'``, ``'R'``, ``'Q'``, ``'M'`` or ``'B'``, optional
        :param quality: Mini-SEED data quality indicator. ``'M'`` and ``'B'``
            (default) are treated the same and indicate best available.
            If ``'B'`` is selected, the output data records will be stamped
            with a ``M``.
        :type minimumlength: float, optional
        :param minimumlength: Enforce minimum segment length - seconds. Only
            time-series segments of this length or longer will be returned.
            .. note:: No data will be returned for selected time windows
                shorter than ``minimumlength``.
        :type longestonly: bool, optional
        :param longestonly: Limit to longest segment only. For each time-series
            selection, only the longest segment is returned. Defaults to
            ``False``.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: :class:`~obspy.core.stream.Stream` or ``None``
        :return: ObsPy Stream object if no ``filename`` is given.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> req = []
        >>> req.append("TA A25A -- BHZ 2010-084T00:00:00 2010-084T00:10:00")
        >>> req.append("TA A25A -- BHN 2010-084T00:00:00 2010-084T00:10:00")
        >>> req.append("TA A25A -- BHE 2010-084T00:00:00 2010-084T00:10:00")
        >>> req = "\\n".join(req) # use only a single backslash!
        >>> print(req)
        TA A25A -- BHZ 2010-084T00:00:00 2010-084T00:10:00
        TA A25A -- BHN 2010-084T00:00:00 2010-084T00:10:00
        TA A25A -- BHE 2010-084T00:00:00 2010-084T00:10:00

        >>> st = client.bulkdataselect(req)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        TA.A25A..BHE | 2010-03-25T00:00:00.000001Z ... | 40.0 Hz, 24001 samples
        TA.A25A..BHN | 2010-03-25T00:00:00.000000Z ... | 40.0 Hz, 24001 samples
        TA.A25A..BHZ | 2010-03-25T00:00:00.000000Z ... | 40.0 Hz, 24000 samples
        """
        url = '/bulkdataselect/query'
        # check for file
        if os.path.isfile(bulk):
            bulk = open(bulk).read()
        # optional parameters
        if quality:
            bulk = "quality %s\n" % (quality.upper()) + bulk
        if minimumlength:
            bulk = "minimumlength %lf\n" % (minimumlength) + bulk
        if longestonly:
            bulk = "longestonly\n" + bulk
        # build up query
        try:
            data = self._fetch(url, data=bulk)
        except HTTPError, e:
            msg = "No waveform data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        # write directly if filename is given
        if filename:
            return self._toFileOrData(filename, data, True)
        # create temporary file for writing data
        tf = NamedTemporaryFile()
        tf.write(data)
        # read stream using obspy.mseed
        tf.seek(0)
        try:
            stream = read(tf.name, 'MSEED')
        except:
            stream = Stream()
        tf.close()
        # remove temporary file:
        try:
            os.remove(tf.name)
        except:
            pass
        return stream

    def availability(self, network="*", station="*", location="*",
                     channel="*", starttime=UTCDateTime() - (60 * 60 * 24 * 7),
                     endtime=UTCDateTime() - (60 * 60 * 24 * 7) + 10,
                     lat=None, lon=None, minradius=None, maxradius=None,
                     minlat=None, maxlat=None, minlon=None, maxlon=None,
                     output="bulkdataselect", restricted=False, filename=None,
                     **kwargs):
        """
        Low-level interface for `availability` Web service of IRIS
        (http://www.iris.edu/ws/availability/) - release 1.2.1 (2012-04-06).

        This method returns information about what time series data is
        available at the IRIS DMC. Users can query for station metadata by
        network, station, channel, location, time and other search criteria.
        Results are may be formated in two formats: ``'bulk'`` or ``'xml'``.

        The 'bulk' formatted information can be passed directly to the
        :meth:`~obspy.iris.client.Client.bulkdataselect()` method.

        The XML format contains station locations as well as channel time range
        for :meth:`~obspy.iris.client.Client.availability()`.

        :type network: str, optional
        :param network: Network code, e.g. ``'IU'``, wildcards allowed.
            Defaults to ``'*'``.
        :type station: str, optional
        :param station: Station code, e.g. ``'ANMO'``, wildcards allowed.
            Defaults to ``'*'``.
        :type location: str, optional
        :param location: Location code, e.g. ``'00'``, wildcards allowed.
            Defaults to ``'*'``.
            Use ``'--'`` for empty location codes.
        :type channel: str, optional
        :param channel: Channel code, e.g. ``'BHZ'``, wildcards allowed.
            Defaults to ``'*'``.
        :type restricted: bool, optional
        :param restricted: If ``True``, availability of restricted as well as
            unrestricted data is reported. Defaults to ``False``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type minlat: float, optional
        :param minlat: Minimum latitude for rectangular bounding box.
        :type minlon: float, optional
        :param minlon: Minimum longitude for rectangular bounding box.
        :type maxlat: float, optional
        :param maxlat: Maximum latitude for rectangular bounding box.
        :type maxlon: float, optional
        :param maxlon: Maximum longitude for rectangular bounding box.
        :type lat: float, optional
        :param lat: Latitude of center point for circular bounding area.
        :type lon: float, optional
        :param lon: Longitude of center point for circular bounding area.
        :type minradius: float, optional
        :param minradius: Minimum radius for circular bounding area.
        :type maxradius: float, optional
        :param maxradius: Maximum radius for circular bounding area.
        :type output: str, optional
        :param output: Output format, either ``"bulkdataselect"`` or ``"xml"``.
            Defaults to ``"bulkdataselect"``.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: str or ``None``
        :return: String that lists available channels, either as plaintext
            `bulkdataselect` format (``output="bulkdataselect"``) or in XML
            format (``output="xml"``) if no ``filename`` is given.

        .. note::

            For restricting data by geographical coordinates either:

            * all of ``minlat``, ``maxlat``, ``minlon`` and ``maxlon`` have to
              be specified for a rectangular bounding box, or
            * all of ``lat``, ``lon``, ``minradius`` and ``maxradius`` have to
              be specified for a circular bounding area

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> t1 = UTCDateTime("2010-02-27T06:30:00")
        >>> t2 = UTCDateTime("2010-02-27T06:40:00")
        >>> result = client.availability("IU", "B*", "*", "BH*", t1, t2)
        >>> print(result)
        IU BBSR 00 BH1 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 00 BH2 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 00 BHZ 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 10 BHE 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 10 BHN 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BBSR 10 BHZ 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BILL 00 BHE 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BILL 00 BHN 2010-02-27T06:30:00 2010-02-27T06:40:00
        IU BILL 00 BHZ 2010-02-27T06:30:00 2010-02-27T06:40:00
        <BLANKLINE>

        >>> st = client.bulkdataselect(result)
        >>> print(st)  # doctest: +ELLIPSIS
        9 Trace(s) in Stream:
        IU.BBSR.00.BH1 | 2010-02-27T06:30:00... | 40.0 Hz, 24000 samples
        IU.BBSR.00.BH2 | 2010-02-27T06:30:00... | 40.0 Hz, 24000 samples
        IU.BBSR.00.BHZ | 2010-02-27T06:30:00... | 40.0 Hz, 24000 samples
        IU.BBSR.10.BHE | 2010-02-27T06:30:00... | 40.0 Hz, 24000 samples
        IU.BBSR.10.BHN | 2010-02-27T06:30:00... | 40.0 Hz, 24000 samples
        IU.BBSR.10.BHZ | 2010-02-27T06:30:00... | 40.0 Hz, 24000 samples
        IU.BILL.00.BHE | 2010-02-27T06:30:00... | 20.0 Hz, 12000 samples
        IU.BILL.00.BHN | 2010-02-27T06:30:00... | 20.0 Hz, 12000 samples
        IU.BILL.00.BHZ | 2010-02-27T06:30:00... | 20.0 Hz, 12000 samples
        """
        url = '/availability/query'
        # build up query
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        try:
            kwargs['starttime'] = UTCDateTime(starttime).formatIRISWebService()
        except:
            kwargs['starttime'] = starttime
        try:
            kwargs['endtime'] = UTCDateTime(endtime).formatIRISWebService()
        except:
            kwargs['endtime'] = endtime
        # stay backward compatible to API change, see #419 and also
        # http://iris.washington.edu/pipermail/webservices/2012-October/000322
        output = str(output).lower()
        if output == 'bulk':
            msg = "output format 'bulk' for Client.availability is " + \
                " deprecated, please use 'bulkdataselect' instead"
            warnings.warn(msg, DeprecationWarning)
            output = 'bulkdataselect'
        kwargs['output'] = output
        kwargs['restricted'] = str(restricted).lower()
        # sanity checking geographical bounding areas
        rectangular = (minlat, minlon, maxlat, maxlon)
        circular = (lon, lat, minradius, maxradius)
        # helper variables to check the user's selection
        any_rectangular = any([value != None for value in rectangular])
        any_circular = any([value != None for value in circular])
        all_rectangular = all([value != None for value in rectangular])
        all_circular = all([value != None for value in circular])
        # not both can be specified at the same time
        if any_rectangular and any_circular:
            msg = "Rectangular and circular bounding areas can not be combined"
            raise ValueError(msg)
        # check and setup rectangular box criteria
        if any_rectangular:
            if not all_rectangular:
                msg = "Missing constraints for rectangular bounding box"
                raise ValueError(msg)
            kwargs['minlat'] = str(minlat)
            kwargs['minlon'] = str(minlon)
            kwargs['maxlat'] = str(maxlat)
            kwargs['maxlon'] = str(maxlon)
        # check and setup circular box criteria
        if any_circular:
            if not all_circular:
                msg = "Missing constraints for circular bounding area"
                raise ValueError(msg)
            kwargs['lat'] = str(lat)
            kwargs['lon'] = str(lon)
            kwargs['minradius'] = str(minradius)
            kwargs['maxradius'] = str(maxradius)
        # checking output options
        if not kwargs['output'] in ("bulkdataselect", "xml"):
            msg = "kwarg output must be either 'bulkdataselect' or 'xml'."
            raise ValueError(msg)
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            if e.code == 404 and e.msg == 'Not Found':
                data = ''
            else:
                raise
        return self._toFileOrData(filename, data)

    def sacpz(self, network, station, location="*", channel="*",
              starttime=None, endtime=None, filename=None, **kwargs):
        """
        Low-level interface for `sacpz` Web service of IRIS
        (http://www.iris.edu/ws/sacpz/) - release 1.1.1 (2012-1-9).

        This method provides access to instrument response information
        (per-channel) as poles and zeros in the ASCII format used by SAC and
        other programs. Users can query for channel response by network,
        station, channel, location and time.

        :type network: str
        :param network: Network code, e.g. ``'IU'``.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'``.
        :type location: str, optional
        :param location: Location code, e.g. ``'00'``, wildcards allowed.
            Defaults to ``'*'``.
        :type channel: str, optional
        :param channel: Channel code, e.g. ``'BHZ'``, wildcards allowed.
            Defaults to ``'*'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime` optional
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: End date and time. Requires starttime parameter.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: str or ``None``
        :return: String with SAC poles and zeros information if no ``filename``
            is given.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> dt = UTCDateTime("2005-01-01")
        >>> sacpz = client.sacpz("IU", "ANMO", "00", "BHZ", dt)
        >>> print(sacpz)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        * **********************************
        * NETWORK   (KNETWK): IU
        * STATION    (KSTNM): ANMO
        * LOCATION   (KHOLE): 00
        * CHANNEL   (KCMPNM): BHZ
        * CREATED           : ...
        * START             : 2002-11-19T21:07:00
        * END               : 2008-06-30T00:00:00
        * DESCRIPTION       : Albuquerque, New Mexico, USA
        * LATITUDE          : 34.945981
        * LONGITUDE         : -106.457133
        * ELEVATION         : 1671.0
        * DEPTH             : 145.0
        * DIP               : 0.0
        * AZIMUTH           : 0.0
        * SAMPLE RATE       : 20.0
        * INPUT UNIT        : M
        * OUTPUT UNIT       : COUNTS
        * INSTTYPE          : Geotech KS-54000 Borehole Seismometer
        * INSTGAIN          : 2.204000e+03 (M/S)
        * COMMENT           :
        * SENSITIVITY       : 9.244000e+08 (M/S)
        * A0                : 8.608300e+04
        * **********************************
        ZEROS    3
            +0.000000e+00    +0.000000e+00
            +0.000000e+00    +0.000000e+00
            +0.000000e+00    +0.000000e+00
        POLES    5
            -5.943130e+01    +0.000000e+00
            -2.271210e+01    +2.710650e+01
            -2.271210e+01    -2.710650e+01
            -4.800400e-03    +0.000000e+00
            -7.319900e-02    +0.000000e+00
        CONSTANT    7.957513e+13
        <BLANKLINE>
        <BLANKLINE>
        <BLANKLINE>
        """
        url = '/sacpz/query'
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        # convert UTCDateTime to string for query
        if starttime and endtime:
            try:
                kwargs['starttime'] = \
                    UTCDateTime(starttime).formatIRISWebService()
            except:
                kwargs['starttime'] = starttime
            try:
                kwargs['endtime'] = UTCDateTime(endtime).formatIRISWebService()
            except:
                kwargs['endtime'] = endtime
        elif starttime:
            try:
                kwargs['time'] = UTCDateTime(starttime).formatIRISWebService()
            except:
                kwargs['time'] = starttime
        data = self._fetch(url, **kwargs)
        return self._toFileOrData(filename, data)

    def distaz(self, stalat, stalon, evtlat, evtlon):
        """
        Low-level interface for `distaz` Web service of IRIS
        (http://www.iris.edu/ws/distaz/) - release 1.0.1 (2010).

        This method will calculate the great-circle angular distance, azimuth,
        and backazimuth between two geographic coordinate pairs. All results
        are reported in degrees, with azimuth and backazimuth measured
        clockwise from North.

        :type stalat: float
        :param stalat: Station latitude.
        :type stalon: float
        :param stalon: Station longitude.
        :type evtlat: float
        :param evtlat: Event latitude.
        :type evtlon: float
        :param evtlon: Event longitude.
        :rtype: dict
        :return: Dictionary containing values for azimuth, backazimuth and
            distance.

        The azimuth is the angle from the station to the event, while the
        backazimuth is the angle from the event to the station.

        Latitudes are converted to geocentric latitudes using the WGS84
        spheroid to correct for ellipticity.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> client = Client()
        >>> result = client.distaz(stalat=1.1, stalon=1.2, evtlat=3.2,
        ...                        evtlon=1.4)
        >>> print(result['distance'])
        2.09554
        >>> print(result['backazimuth'])
        5.46946
        >>> print(result['azimuth'])
        185.47692
        """
        # set JSON as expected content type
        headers = {'Accept': 'application/json'}
        # build up query
        url = '/distaz/query'
        try:
            data = self._fetch(url, headers=headers, stalat=stalat,
                               stalon=stalon, evtlat=evtlat, evtlon=evtlon)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        data = json.loads(data)
        results = {}
        results['distance'] = data['distance']
        results['backazimuth'] = data['backAzimuth']
        results['azimuth'] = data['azimuth']
        return results

    def flinnengdahl(self, lat, lon, rtype="both"):
        """
        Low-level interface for `flinnengdahl` Web service of IRIS
        (http://www.iris.edu/ws/flinnengdahl/) - release 1.1 (2011-06-08).

        This method converts a latitude, longitude pair into either a
        `Flinn-Engdahl <http://en.wikipedia.org/wiki/Flinn-Engdahl_regions>`_
        seismic region code or region name.

        :type lat: float
        :param lat: Latitude of interest.
        :type lon: float
        :param lon: Longitude of interest.
        :type rtype: ``'code'``, ``'region'`` or ``'both'``
        :param rtype: Return type. Defaults to ``'both'``.
        :rtype: int, str, or tuple
        :returns: Returns Flinn-Engdahl region code or name or both, depending
            on the request type parameter ``rtype``.

        .. rubric:: Examples

        >>> from obspy.iris import Client
        >>> client = Client()
        >>> client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="code")
        683

        >>> client.flinnengdahl(lat=42, lon=-122.24, rtype="region")
        'OREGON'

        >>> client.flinnengdahl(lat=-20.5, lon=-100.6)
        (683, 'SOUTHEAST CENTRAL PACIFIC OCEAN')
        """
        url = '/flinnengdahl/%s?lat=%f&lon=%f'
        # check rtype
        try:
            if rtype == 'code':
                return int(self._fetch(url % (rtype, lat, lon)))
            elif rtype == 'region':
                return self._fetch(url % (rtype, lat, lon)).strip()
            else:
                code = int(self._fetch(url % ('code', lat, lon)))
                region = self._fetch(url % ('region', lat, lon)).strip()
                return (code, region)
        except HTTPError, e:
            msg = "No Flinn-Engdahl data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)

    def traveltime(self, model='iasp91', phases=DEFAULT_PHASES, evdepth=0.0,
                   distdeg=None, distkm=None, evloc=None, staloc=None,
                   noheader=False, traveltimeonly=False, rayparamonly=False,
                   mintimeonly=False, filename=None):
        """
        Low-level interface for `traveltime` Web service of IRIS
        (http://www.iris.edu/ws/traveltime/) - release 1.1.1 (2012-05-15).

        This method will calculates travel-times for seismic phases using a 1-D
        spherical earth model.

        :type model: str, optional
        :param model: Name of 1-D earth velocity model to be used. Available
            models include:

                * ``'iasp91'`` (default) - by Int'l Assoc of Seismology and
                  Physics of the Earth's Interior
                * ``'prem'`` - Preliminary Reference Earth Model
                * ``'ak135'``
        :type phases: list of str, optional
        :param phases: Comma separated list of phases. The default is as
            follows::
                ['p','s','P','S','Pn','Sn','PcP','ScS','Pdiff','Sdiff',
                 'PKP','SKS','PKiKP','SKiKS','PKIKP','SKIKS']
            Invalid phases will be ignored. Valid arbitrary phases can be made
            up e.g. sSKJKP. See
            `TauP documentation <http://www.seis.sc.edu/TauP/>`_ for more
            information.
        :type evdepth: float, optional
        :param evdepth: The depth of the event, in kilometers. Default is ``0``
            km.

        **Geographical Parameters - required**

        The travel time web service requires a great-circle distance between an
        event and station be specified. There are three methods of specifying
        this distance:

        * Specify a great-circle distance in degrees, using ``distdeg``
        * Specify a great-circle distance in kilometers, using ``distkm``
        * Specify an event location and one or more station locations,
          using ``evloc`` and ``staloc``

        :type distdeg: float or list of float, optional
        :param evtlon: Great-circle distance from source to station, in decimal
            degrees. Multiple distances may be specified as a list.
        :type distkm: float or list of float, optional
        :param distkm: Distance between the source and station, in kilometers.
            Multiple distances may be specified as a list.
        :type evloc: tuple of two floats, optional
        :param evloc: The Event location (lat,lon) using decimal degrees.
        :type staloc: tuple of two floats or list of tuples, optional
        :param staloc: Station locations for which the phases will be listed.
            The general format is (lat,lon). Specify multiple station locations
            with a list, e.g. ``[(lat1,lon1),(lat2,lon2),...,(latn,lonn)]``.

        **Output Parameters**

        :type noheader: bool, optional
        :param noheader: Specifying noheader will strip the header from the
            resulting table. Defaults to ``False``.
        :type traveltimeonly: bool, optional
        :param traveltimeonly: Returns a space-separated list of travel
            times, in seconds. Defaults to ``False``.

            .. note:: Travel times are produced in ascending order regardless
                of the order in which the phases are specified
        :type rayparamonly: bool, optional
        :param rayparamonly: Returns a space-separated list of ray parameters,
            in sec/deg.. Defaults to ``False``.
        :type mintimeonly: bool, optional
        :param mintimeonly: Returns only the first arrival of each phase for
            each distance. Defaults to ``False``.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: str or ``None``
        :return: ASCII travel time table if no ``filename`` is given.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> client = Client()
        >>> result = client.traveltime(evloc=(-36.122,-72.898),
        ...     staloc=[(-33.45,-70.67),(47.61,-122.33),(35.69,139.69)],
        ...     evdepth=22.9)
        >>> print(result)  # doctest: +ELLIPSIS  +NORMALIZE_WHITESPACE
        Model: iasp91
        Distance   Depth   Phase   Travel    Ray Param  Takeoff  Incident ...
          (deg)     (km)   Name    Time (s)  p (s/deg)   (deg)    (deg)   ...
        ------------------------------------------------------------------...
            3.24    22.9   P         49.39    13.749     53.77    45.82   ...
            3.24    22.9   Pn        49.40    13.754     53.80    45.84   ...
        """
        kwargs = {}
        kwargs['model'] = str(model)
        kwargs['phases'] = ','.join([str(p) for p in list(phases)])
        kwargs['evdepth'] = float(evdepth)
        if distdeg:
            kwargs['distdeg'] = \
                ','.join([str(float(d)) for d in list(distdeg)])
        elif distkm:
            kwargs['distkm'] = ','.join([str(float(d)) for d in list(distkm)])
        elif evloc and staloc:
            if not isinstance(evloc, tuple):
                raise TypeError("evloc needs to be a tuple")
            kwargs['evloc'] = \
                "[%s]" % (','.join([str(float(n)) for n in evloc]))
            if isinstance(staloc, tuple):
                # single station coordinates
                staloc = [staloc]
            if len(staloc) == 0:
                raise ValueError("staloc needs to be set if using evloc")
            temp = ''
            for loc in staloc:
                if not isinstance(loc, tuple):
                    msg = "staloc needs to be a tuple or list of tuples"
                    raise TypeError(msg)
                temp += ",[%s]" % (','.join([str(float(n)) for n in loc]))
            kwargs['staloc'] = temp[1:]
        else:
            msg = "Missing or incorrect geographical parameters distdeg, " + \
                "distkm or evloc/staloc."
            raise ValueError(msg)
        if noheader:
            kwargs['noheader'] = 1
        elif traveltimeonly:
            kwargs['traveltimeonly'] = 1
        elif rayparamonly:
            kwargs['rayparamonly'] = 1
        elif mintimeonly:
            kwargs['mintimeonly'] = 1
        # build up query
        url = '/traveltime/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return self._toFileOrData(filename, data)

    def evalresp(self, network, station, location, channel, time=UTCDateTime(),
                 minfreq=0.00001, maxfreq=None, nfreq=200, units='def',
                 width=800, height=600, annotate=True, output='plot',
                 filename=None, **kwargs):
        """
        Low-level interface for `evalresp` Web service of IRIS
        (http://www.iris.edu/ws/evalresp/) - release 1.0.0 (2011-08-11).

        This method evaluates instrument response information stored at the
        IRIS DMC and outputs ASCII data or
        `Bode Plots <http://en.wikipedia.org/wiki/Bode_plots>`_.

        :type network: str
        :param network: Network code, e.g. ``'IU'``.
        :type station: str
        :param station: Station code, e.g. ``'ANMO'``.
        :type location: str
        :param location: Location code, e.g. ``'00'``. Use ``'--'`` for empty
            location codes.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``.
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Evaluate the response at the given time. If not specified,
            the current time is used.
        :type minfreq: float, optional
        :param minfreq: The minimum frequency (Hz) at which response will be
            evaluated. Must be positive and less than the ``maxfreq`` value.
            Defaults to ``0.00001`` Hz (1/day ~ 0.000012 Hz).
        :type maxfreq: float, optional
        :param maxfreq: The maximum frequency (Hz) at which response will be
            evaluated. Must be positive and greater than the ``minfreq`` value.
            Defaults to the channel sample-rate or the frequency of
            sensitivity, which ever is larger.
        :type nfreq: int, optional
        :param nfreq: Number frequencies at which response will be evaluated.
            Must be a positive integer no greater than ``10000``. The
            instrument response is evaluated on a equally spaced logarithmic
            scale. Defaults to ``200``.
        :type units:  ``'def'``, ``'dis'``, ``'vel'``, ``'acc'``, optional
        :param units: Output Unit. Defaults to ``'def'``.

            ``'def'``
                default units indicated in response metadata
            ``'dis'``
                converts to units of displacement
            ``'vel'``
                converts to units of velocity
            ``'acc'``
                converts to units of acceleration

            If units are not specified, then the units will default to those
            indicated in the response metadata
        :type width: int, optional
        :param width: The width of the generated plot. Defaults to ``800``.
            Can only be used with the ``output='plot'``, ``output='plot-amp'``
            and ``output='plot-phase'`` options. Cannot be larger than ``5000``
            and the product of width and height cannot be larger than
            ``6,000,000``.
        :type height: int, optional
        :param height: The height of the generated plot. Defaults to ``600``.
            Can only be used with the ``output='plot'``, ``output='plot-amp'``
            and ``output='plot-phase'`` options. Cannot be larger than ``5000``
            and the product of width and height cannot be larger than
            ``6,000,000``.
        :type annotate: bool, optional
        :param annotate: Can be either ``True`` or ``False``. Defaults
            to ``True``.

            * Draws vertical lines at the Nyquist frequency (one half the
              sample rate).
            * Draw a vertical line at the stage-zero frequency of sensitivity.
            * Draws a horizontal line at the stage-zero gain.

            Can only be used with the ``output='plot'``, ``output='plot-amp'``
            and ``output='plot-phase'`` options.
        :type output: str
        :param output: Output Options. Defaults to ``'plot'``.

            ``'fap'``
                Three column ASCII (frequency, amplitude, phase)
            ``'cs'``
                Three column ASCII (frequency, real, imaginary)
            ``'plot'``
                Amplitude and phase plot
            ``'plot-amp'``
                Amplitude only plot
            ``'plot-phase'``
                Phase only plot

            Plots are stored to the file system if the parameter ``filename``
            is set, otherwise it will try to use matplotlib to directly plot
            the returned image.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype: numpy.ndarray, str or `None`
        :returns: Returns either a NumPy :class:`~numpy.ndarray`, image string
            or nothing, depending on the ``output`` parameter.

        .. rubric:: Examples

        (1) Returning frequency, amplitude, phase of first point.

            >>> from obspy.iris import Client
            >>> client = Client()
            >>> dt = UTCDateTime("2005-01-01")
            >>> data = client.evalresp("IU", "ANMO", "00", "BHZ", dt,
            ...                        output='fap')
            >>> data[0]  # frequency, amplitude, phase of first point
            array([  1.00000000e-05,   1.20280200e+04,   1.79200700e+02])

        (2) Returning amplitude and phase plot.

            >>> from obspy.iris import Client
            >>> client = Client()
            >>> dt = UTCDateTime("2005-01-01")
            >>> client.evalresp("IU", "ANMO", "00", "BHZ", dt) # doctest: +SKIP

            .. plot::

                from obspy import UTCDateTime
                from obspy.iris import Client
                client = Client()
                dt = UTCDateTime("2005-01-01")
                client.evalresp("IU", "ANMO", "00", "BHZ", dt)
        """
        url = '/evalresp/query'
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        try:
            kwargs['time'] = UTCDateTime(time).formatIRISWebService()
        except:
            kwargs['time'] = time
        kwargs['minfreq'] = float(minfreq)
        if maxfreq:
            kwargs['maxfreq'] = float(maxfreq)
        kwargs['nfreq'] = int(nfreq)
        if units in ['def', 'dis', 'vel', 'acc']:
            kwargs['units'] = units
        else:
            kwargs['units'] = 'def'
        if output in ['fap', 'cs', 'plot', 'plot-amp', 'plot-phase']:
            kwargs['output'] = output
        else:
            kwargs['output'] = 'plot'
        # height, width and annotate work only for plots
        if 'plot' in output:
            kwargs['width'] = int(width)
            kwargs['height'] = int(height)
            kwargs['annotate'] = bool(annotate)
        data = self._fetch(url, **kwargs)
        # check output
        if 'plot' in output:
            # image
            if filename is None:
                # ugly way to show an image
                from matplotlib import image
                import matplotlib.pyplot as plt
                # need temporary file for reading into matplotlib
                tf = NamedTemporaryFile()
                tf.write(data)
                tf.close()
                # create new figure
                fig = plt.figure()
                # new axes using full window
                ax = fig.add_axes([0, 0, 1, 1])
                # force matplotlib to use internal PNG reader. image.imread
                # will use PIL if available
                img = image._png.read_png(tf.name)
                # add image to axis
                ax.imshow(img)
                # delete temporary file
                os.remove(tf.name)
                # hide axes
                ax.axison = False
                # show plot
                plt.show()
            else:
                self._toFileOrData(filename, data, binary=True)
        else:
            # ASCII data
            if filename is None:
                return loadtxt(StringIO.StringIO(data), ndlim=1)
            else:
                return self._toFileOrData(filename, data)

    def event(self, filename=None, **kwargs):
        """
        Low-level interface for `event` Web service of IRIS
        (http://www.iris.edu/ws/event/) - release 1.2.1 (2012-02-29).

        This method returns contributed earthquake origin and magnitude
        estimates stored in the IRIS database. Selected information is returned
        in `QuakeML <https://quake.ethz.ch/quakeml/>`_ format.

        The IRIS DMC receives earthquake location and magnitude information
        primarily from the
        `USGS NEIC <http://earthquake.usgs.gov/regional/neic/>`_ and the
        `ISC <http://www.isc.ac.uk/>`_, other sources include the
        `Global CMT project <http://www.globalcmt.org/>`_ and the
        `USArray ANF <http://anf.ucsd.edu/>`_.

        **Geographic constraints - bounding rectangle**

        The following four parameters work together to specify a boundary
        rectangle. All four parameters are optional, but they may not be mixed
        with the parameters used for searching within a defined radius.

        :type minlat: float, optional
        :param minlat: Specify the southern boundary. The minimum latitude must
            be between -90 and 90 degrees inclusive (and less than or equal to
            maxlat). If not specified, then this value defaults to ``-90``.
        :type maxlat: float, optional
        :param maxlat: Specify the northern boundary. The maximum latitude must
            be between -90 and 90 degrees inclusive and greater than or equal
            to minlat. If not specified, then this value defaults to ``90``.
        :type minlon: float, optional
        :param minlon: Specify the western boundary. The minimum longitude must
            be between -180 and 180 degrees inclusive. If not specified, then
            this value defaults to ``-180``. If minlon > maxlon, then the
            boundary will cross the -180/180 meridian
        :type maxlon: float, optional
        :param maxlon: Specify the eastern boundary. The minimum longitude must
            be between -180 and 180 degrees inclusive. If not specified, then
            this value defaults to +180. If maxlon < minlon, then the boundary
            will cross the -180/180 meridian

        **Geographic constraints - bounding radius**

        The following four parameters work together to specify a boundary using
        a radius around a coordinate. ``lat``, ``lon``, and ``maxradius`` are
        all required, and must be used together. These parameters are
        incompatible with the boundary-box parameters described above.

        :type lat: float, optional
        :param lat: Specify the central latitude point, in degrees. This value
            must be between -90 and 90 degrees. This MUST be used in
            conjunction with the lon and maxradius parameters.
        :type lon: float, optional
        :param lon: Specify the central longitude point, in degrees. This MUST
            be used in conjunction with the lat and maxradius parameters.
        :type maxradius: float, optional
        :param maxradius: Specify the maximum radius, in degrees. Only
            earthquakes within maxradius degrees of the lat/lon point will be
            retrieved. This MUST be used in conjunction with the lat and lon
            parameters.
        :type minradius: float, optional
        :param minradius: This optional parameter allows for the exclusion of
            events that are closer than minradius degrees from the specified
            lat/lon point. This MUST be used in conjunction with the lat, lon,
            and maxradius parameters and is subject to the same restrictions.
            If this parameter isn't specified, then it defaults to ``0.0``
            degrees.

        **Depth constraints**

        :type mindepth: float, optional
        :param mindepth: Specify minimum depth (kilometers), values increase
            positively with depth, e.g. ``-1``.
        :type maxdepth: float, optional
        :param maxdepth: Specify maximum depth (kilometers), values increase
            positively with depth, e.g. ``20``.

        **Temporal constraints**

        The following two parameters impose time constrants on the query.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Limit results to the events occurring after the
            specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Limit results to the events occurring before the
            specified end time.

        **Magnitude constraints**

        :type minmag: float, optional
        :param minmag: Specify a minimum magnitude.
        :type maxmag: float, optional
        :param maxmag: Specify a maximum magnitude.
        :type magtype: string, optional
        :param magtype: Specify magnitude type. Some common types (there are
            many) include ``"Ml"`` (local/Richter magnitude), ``"Ms"`` (surface
            magnitude), ``"mb"`` (body wave magnitude), ``"Mw"`` (moment
            magnitude).

        **Subset the data by organization**

        :type catalog: string, optional
        :param catalog: Specify a catalog
            [`available catalogs <http://www.iris.edu/ws/event/catalogs>`_].
            Results will include any origins which contain the specified
            catalog text, i.e. ``"PDE"`` will match ``"NEIC PDE"``
        :type contributor: string, optional
        :param contributor: Specify a contributor [`available
            contributors <http://www.iris.edu/ws/event/contributors>`_]. When
            selecting a contributor, the result includes a preferred origin as
            specified by the contributor. Results will include any origins
            which contain the specified contributor text, i.e. ``"NEIC"`` will
            match ``"NEIC PDE-Q"``.

        **Specifying an event using an IRIS ID number**

        Individual events can be retrieved using ID numbers assigned by IRIS.
        When these parameters are used, then only the ``includeallmagnitudes``
        and ``preferredonly`` parameters are also allowed.

        :type eventid: int, optional
        :param eventid: Retrieve an event based on the unique IRIS event id.
        :type originid: int, optional
        :param originid: Retrieve an event based on the unique IRIS origin id.
        :type magnitudeid: int, optional
        :param magnitudeid: Retrieve an event based on the unique IRIS
            magnitude id.

        **Miscellaneous parameters**

        These parameters affect how the search is conducted, and how the events
        are returned.

        :type limit: int, optional
        :param limit: Limit the results to the specified number of events. This
            value must be 10 or greater. By default, the results are not
            limited.
        :type orderby: ``"time"`` or ``"magnitude"``, optional
        :param orderby: Sort the resulting events in order of descending time
            or magnitude. By default, results are sorted in descending time
            order.
        :type updatedafter:  :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param updatedafter: Select origins updated after a certain date. This
            is most useful for synchronization purposes.
        :type includeallmagnitudes: bool, optional
        :param includeallmagnitudes: This will include all the magnitudes in
            search and print criteria. If magnitudes do not exist for a certain
            origin, the search algorithm will consider it a miss and therefore
            will not include the event. Defaults to ``True``.
        :type includearrivals: bool, optional
        :param includearrivals: If this event has associated phase arrival
            information, then it will be included in the results. Defaults to
            ``False``.
        :type preferredonly: bool, optional
        :param preferredonly: Include preferred estimates only. When catalog is
            selected, the result returned will include the preferred origin as
            specified by the catalog. Defaults to ``True``.
        :type filename: str, optional
        :param filename: Name of a output file. If this parameter is given
            nothing will be returned. Default is ``None``.
        :rtype:  str or ``None``
        :return: QuakeML formated string if no ``filename`` is given.

        .. rubric:: Example

        >>> from obspy.iris import Client
        >>> client = Client()
        >>> events = client.event(minmag=9.1)
        >>> print(events)  # doctest: +ELLIPSIS
        <q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" ...
        """
        # convert UTCDateTimes to string
        try:
            kwargs['starttime'] = \
                UTCDateTime(kwargs['starttime']).formatIRISWebService()
        except KeyError:
            pass
        try:
            kwargs['endtime'] = \
                UTCDateTime(kwargs['endtime']).formatIRISWebService()
        except KeyError:
            pass
        try:
            kwargs['updatedafter'] = \
                str(UTCDateTime(kwargs['updatedafter']).date)
        except KeyError:
            pass
        # convert boolean values to string
        if 'includeallmagnitudes' in kwargs:
            if not kwargs['includeallmagnitudes']:
                kwargs['includeallmagnitudes'] = 'no'
            else:
                kwargs['includeallmagnitudes'] = 'yes'
        if 'includearrivals' in kwargs:
            if not kwargs['includearrivals']:
                kwargs['includearrivals'] = 'no'
            else:
                kwargs['includearrivals'] = 'yes'
        if 'preferredonly' in kwargs:
            if not kwargs['preferredonly']:
                kwargs['preferredonly'] = 'no'
            else:
                kwargs['preferredonly'] = 'yes'
        # build up query
        url = '/event/query'
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError, e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return self._toFileOrData(filename, data)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
