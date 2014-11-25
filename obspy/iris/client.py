# -*- coding: utf-8 -*-
"""
IRIS Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str
from future import standard_library
with standard_library.hooks():
    import urllib.parse
    import urllib.request

from obspy import UTCDateTime, read, Stream, __version__
from obspy.core.util import NamedTemporaryFile, loadtxt

import io
import json
import platform


DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (__version__,
                                                   platform.platform(),
                                                   platform.python_version())
DEFAULT_PHASES = ['p', 's', 'P', 'S', 'Pn', 'Sn', 'PcP', 'ScS', 'Pdiff',
                  'Sdiff', 'PKP', 'SKS', 'PKiKP', 'SKiKS', 'PKIKP', 'SKIKS']
DEPR_WARN = ("This service was shut down on the server side in December "
             "2013, please use %s instead. Further information: "
             "http://www.iris.edu/ds/nodes/dmc/news/2013/03/"
             "new-fdsn-web-services-and-retirement-of-deprecated-services/")
DEPR_WARNS = dict([(new, DEPR_WARN % "obspy.fdsn.client.Client.%s" % new)
                   for new in ["get_waveform", "get_events", "get_stations",
                               "get_waveform_bulk"]])
DEFAULT_SERVICE_VERSIONS = {"timeseries": 1, "sacpz": 1, "resp": 1,
                            "evalresp": 1, "traveltime": 1, "flinnengdahl": 2,
                            "distaz": 1}


class Client(object):
    """
    IRIS Web service request client.

    :type base_url: str, optional
    :param base_url: Base URL of the IRIS Web service (default
        is ``'http://service.iris.edu/irisws'``).
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
    :type major_versions: dict
    :param major_versions: Allows to specify custom major version numbers
        for individual services (e.g.
        `major_versions={'evalresp': 2, 'sacpz': 3}`), otherwise the
        latest version at time of implementation will be used.

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
    def __init__(self, base_url="http://service.iris.edu/irisws",
                 user="", password="", timeout=20, debug=False,
                 user_agent=DEFAULT_USER_AGENT, major_versions={}):
        """
        Initializes the IRIS Web service client.

        See :mod:`obspy.iris` for all parameters.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.debug = debug
        self.user_agent = user_agent
        self.major_versions = DEFAULT_SERVICE_VERSIONS
        self.major_versions.update(major_versions)
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(auth_handler)
        # install globally
        urllib.request.install_opener(opener)

    def _fetch(self, service, data=None, headers={}, param_list=[], **params):
        """
        Send a HTTP request via urllib2.

        :type service: str
        :param service: Name of service
        :type data: str
        :param data: Channel list as returned by `availability` Web service
        :type headers: dict, optional
        :param headers: Additional header information for request
        """
        headers['User-Agent'] = self.user_agent
        # replace special characters
        remoteaddr = "/".join([self.base_url.rstrip("/"), service,
                               str(self.major_versions[service]), "query"])
        options = '&'.join(param_list)
        if params:
            if options:
                options += '&'
            options += urllib.parse.urlencode(params)
        if options:
            remoteaddr = "%s?%s" % (remoteaddr, options)
        if self.debug:
            print('\nRequesting %s' % (remoteaddr))
        req = urllib.request.Request(url=remoteaddr, data=data,
                                     headers=headers)
        response = urllib.request.urlopen(req, timeout=self.timeout)
        doc = response.read()
        return doc

    def _toFileOrData(self, filename, data, binary=False):
        """
        Either writes data into a file if filename is given or directly returns
        it.

        :type filename: str or file
        :param filename: File or object being written to. If None, a string
            will be returned.
        :type data: str or bytes
        :param data: The data being written or returned.
        :type binary: bool, optional
        :param binary: Whether to write the data as binary or text. Defaults to
            binary.
        """
        if filename is None:
            return data
        if binary:
            method = 'wb'
        else:
            method = 'wt'
        file_opened = False
        # file name is given, create fh, write to file and return nothing
        if hasattr(filename, "write") and callable(filename.write):
            fh = filename
        elif isinstance(filename, (str, native_str)):
            fh = open(filename, method)
            file_opened = True
        else:
            msg = ("Parameter 'filename' must be either a string or an open "
                   "file-like object.")
            raise TypeError(msg)
        try:
            fh.write(data if binary else data.decode('utf-8'))
        finally:
            # Only close if also opened.
            if file_opened is True:
                fh.close()

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime, quality='B'):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_waveform'])

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, quality='B'):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_waveform'])

    def saveResponse(self, filename, network, station, location, channel,
                     starttime, endtime, format='RESP'):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_stations'])

    def getEvents(self, format='catalog', **kwargs):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_events'])

    def timeseries(self, network, station, location, channel,
                   starttime, endtime, filter=[], filename=None,
                   output='miniseed', **kwargs):
        """
        Low-level interface for `timeseries` Web service of IRIS
        (http://service.iris.edu/irisws/timeseries/)- release 1.3.5
        (2012-06-07).

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

        The following parameters act as filters upon the time series.

        :type filter: list of str, optional
        :param filter: Filter list.  List order matters because each filter
            operation is performed in the order given. For example
            ``filter=["demean", "lp=2.0"]`` will demean and then apply a
            low-pass filter, while ``filter=["lp=2.0", "demean"]`` will apply
            the low-pass filter first, and then demean.

            ``"taper=WIDTH,TYPE"``
                Apply a time domain symmetric tapering function to the
                time series data. The width is specified as a fraction of the
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
                IRIS MiniSEED format
            ``'plot'``
                A simple plot of the time series
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
        [ -1.57488682e-06  -1.26318002e-06  -7.84807128e-07 ...
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
        try:
            data = self._fetch("timeseries", param_list=filter, **kwargs)
        except urllib.request.HTTPError as e:
            msg = "No waveform data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        # write directly if file name is given
        if filename:
            return self._toFileOrData(filename, data, True)
        # create temporary file for writing data
        with NamedTemporaryFile() as tf:
            tf.write(data)
            # read stream using obspy.mseed
            tf.seek(0)
            try:
                stream = read(tf.name, 'MSEED')
            except:
                stream = Stream()
        return stream

    def resp(self, network, station, location="*", channel="*",
             starttime=None, endtime=None, filename=None, **kwargs):
        """
        Low-level interface for `resp` Web service of IRIS
        (http://service.iris.edu/irisws/resp/) - 1.4.1 (2011-04-14).

        This method provides access to channel response information in the SEED
        `RESP <http://www.iris.edu/ds/nodes/dmc/kb/questions/60/>`_
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

        The following three parameters impose time constraints on the query.
        Time may be requested through the use of either time OR the start and
        end times. If no time is specified, then the current time is assumed.

        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param time: Find the response for the given time. Time cannot be used
            with ``starttime`` or ``endtime`` parameters
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Start time, may be used in conjunction with
            ``endtime``.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: End time, may be used in conjunction with
            ``starttime``.
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
        >>> print(data.decode())  # doctest: +ELLIPSIS
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
        try:
            data = self._fetch("resp", **kwargs)
        except urllib.request.HTTPError as e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return self._toFileOrData(filename, data)

    def station(self, network, station, location="*", channel="*",
                starttime=None, endtime=None, level='sta', filename=None,
                **kwargs):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_stations'])

    def dataselect(self, network, station, location, channel,
                   starttime, endtime, quality='B', filename=None, **kwargs):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_waveform'])

    def bulkdataselect(self, bulk, quality=None, filename=None,
                       minimumlength=None, longestonly=False):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/
        new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_waveform_bulk'])

    def availability(self, network="*", station="*", location="*",
                     channel="*", starttime=UTCDateTime() - (60 * 60 * 24 * 7),
                     endtime=UTCDateTime() - (60 * 60 * 24 * 7) + 10,
                     lat=None, lon=None, minradius=None, maxradius=None,
                     minlat=None, maxlat=None, minlon=None, maxlon=None,
                     output="bulkdataselect", restricted=False, filename=None,
                     **kwargs):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_stations'])

    def sacpz(self, network, station, location="*", channel="*",
              starttime=None, endtime=None, filename=None, **kwargs):
        """
        Low-level interface for `sacpz` Web service of IRIS
        (http://service.iris.edu/irisws/sacpz/) - release 1.1.1 (2012-1-9).

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
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: End date and time. Requires ``starttime`` parameter.
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
        >>> print(sacpz.decode())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
        * INSTGAIN          : 1.935000e+03 (M/S)
        * COMMENT           :
        * SENSITIVITY       : 8.115970e+08 (M/S)
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
        CONSTANT    6.986470e+13
        <BLANKLINE>
        <BLANKLINE>
        <BLANKLINE>
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
        elif starttime:
            try:
                kwargs['time'] = UTCDateTime(starttime).formatIRISWebService()
            except:
                kwargs['time'] = starttime
        data = self._fetch("sacpz", **kwargs)
        return self._toFileOrData(filename, data)

    def distaz(self, stalat, stalon, evtlat, evtlon):
        """
        Low-level interface for `distaz` Web service of IRIS
        (http://service.iris.edu/irisws/distaz/) - release 1.0.1 (2010).

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
        try:
            data = self._fetch("distaz", headers=headers, stalat=stalat,
                               stalon=stalon, evtlat=evtlat, evtlon=evtlon)
        except urllib.request.HTTPError as e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        data = json.loads(data.decode())
        results = {}
        results['distance'] = data['distance']
        results['backazimuth'] = data['backAzimuth']
        results['azimuth'] = data['azimuth']
        return results

    def flinnengdahl(self, lat, lon, rtype="both"):
        """
        Low-level interface for `flinnengdahl` Web service of IRIS
        (http://service.iris.edu/irisws/flinnengdahl/) - release 1.1
        (2011-06-08).

        This method converts a latitude, longitude pair into either a
        `Flinn-Engdahl <http://en.wikipedia.org/wiki/Flinn-Engdahl_regions>`_
        seismic region code or region name.

        :type lat: float
        :param lat: Latitude of interest.
        :type lon: float
        :param lon: Longitude of interest.
        :type rtype: str, optional
        :param rtype: Return type. Can be one of ``'code'``, ``'region'`` or
            ``'both'``. Defaults to ``'both'``.
        :rtype: int, str, or tuple
        :returns: Returns Flinn-Engdahl region code or name or both, depending
            on the request type parameter ``rtype``.

        .. rubric:: Examples

        >>> from obspy.iris import Client
        >>> client = Client()
        >>> client.flinnengdahl(lat=-20.5, lon=-100.6, rtype="code")
        683

        >>> print(client.flinnengdahl(lat=42, lon=-122.24, rtype="region"))
        OREGON

        >>> code, region = client.flinnengdahl(lat=-20.5, lon=-100.6)
        >>> print(code, region)
        683 SOUTHEAST CENTRAL PACIFIC OCEAN
        """
        service = 'flinnengdahl'
        # check rtype
        try:
            if rtype == 'code':
                param_list = ["output=%s" % rtype, "lat=%s" % lat,
                              "lon=%s" % lon]
                return int(self._fetch(service, param_list=param_list))
            elif rtype == 'region':
                param_list = ["output=%s" % rtype, "lat=%s" % lat,
                              "lon=%s" % lon]
                return self._fetch(service,
                                   param_list=param_list).strip().decode()
            else:
                param_list = ["output=code", "lat=%s" % lat,
                              "lon=%s" % lon]
                code = int(self._fetch(service, param_list=param_list))
                param_list = ["output=region", "lat=%s" % lat,
                              "lon=%s" % lon]
                region = self._fetch(service, param_list=param_list).strip()
                return (code, region.decode())
        except urllib.request.HTTPError as e:
            msg = "No Flinn-Engdahl data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)

    def traveltime(self, model='iasp91', phases=DEFAULT_PHASES, evdepth=0.0,
                   distdeg=None, distkm=None, evloc=None, staloc=None,
                   noheader=False, traveltimeonly=False, rayparamonly=False,
                   mintimeonly=False, filename=None):
        """
        Low-level interface for `traveltime` Web service of IRIS
        (http://service.iris.edu/irisws/traveltime/) - release 1.1.1
        (2012-05-15).

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
        >>> print(result.decode())  # doctest: +ELLIPSIS  +NORMALIZE_WHITESPACE
        Model: iasp91
        Distance   Depth   Phase   Travel    Ray Param  Takeoff  Incident ...
          (deg)     (km)   Name    Time (s)  p (s/deg)   (deg)    (deg)   ...
        ------------------------------------------------------------------...
            3.24    22.9   P         49.39    13.749     53.77    45.82   ...
            3.24    22.9   Pn        49.40    13.754     53.80    45.84   ...
        """
        kwargs = {}
        kwargs['model'] = str(model)
        kwargs['phases'] = ','.join([str(p) for p in phases])
        kwargs['evdepth'] = float(evdepth)
        if distdeg:
            kwargs['distdeg'] = \
                ','.join([str(float(d)) for d in distdeg])
        elif distkm:
            kwargs['distkm'] = ','.join([str(float(d)) for d in distkm])
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
        try:
            data = self._fetch("traveltime", **kwargs)
        except urllib.request.HTTPError as e:
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
        (http://service.iris.edu/irisws/evalresp/) - release 1.0.0
        (2011-08-11).

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
        :type units:  str, optional
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
        :rtype: :class:`numpy.ndarray`, str or `None`
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
            array([  1.00000000e-05,   1.05599900e+04,   1.79200700e+02])

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
        data = self._fetch("evalresp", **kwargs)
        # check output
        if 'plot' in output:
            # image
            if filename is None:
                # ugly way to show an image
                from matplotlib import image
                import matplotlib.pyplot as plt
                # create new figure
                fig = plt.figure()
                # new axes using full window
                ax = fig.add_axes([0, 0, 1, 1])
                # need temporary file for reading into matplotlib
                with NamedTemporaryFile() as tf:
                    tf.write(data)
                    # force matplotlib to use internal PNG reader. image.imread
                    # will use PIL if available
                    img = image._png.read_png(native_str(tf.name))
                # add image to axis
                ax.imshow(img)
                # hide axes
                ax.axison = False
                # show plot
                plt.show()
            else:
                self._toFileOrData(filename, data, binary=True)
        else:
            # ASCII data
            if filename is None:
                return loadtxt(io.BytesIO(data), ndmin=1)
            else:
                return self._toFileOrData(filename, data, binary=True)

    def event(self, filename=None, **kwargs):
        """
        SHUT DOWN ON SERVER SIDE!

        This service was shut down on the server side in December
        2013, please use :mod:`obspy.fdsn` instead.

        Further information:
        http://www.iris.edu/ds/nodes/dmc/news/2013/03/\
new-fdsn-web-services-and-retirement-of-deprecated-services/
        """
        raise Exception(DEPR_WARNS['get_events'])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
