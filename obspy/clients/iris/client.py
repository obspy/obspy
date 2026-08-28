# -*- coding: utf-8 -*-
"""
EarthScope (former IRIS) Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
import platform
import urllib.request as urllib_request
from lxml import objectify
from urllib.parse import urlencode

import numpy as np

from obspy import Stream, UTCDateTime, __version__, read
from obspy.core.util.decorator import deprecated
from obspy.core.util import NamedTemporaryFile


DEFAULT_USER_AGENT = "ObsPy/%s (%s, Python %s)" % (__version__,
                                                   platform.platform(),
                                                   platform.python_version())
DEFAULT_PHASES = ['p', 's', 'P', 'S', 'Pn', 'Sn', 'PcP', 'ScS', 'Pdiff',
                  'Sdiff', 'PKP', 'SKS', 'PKiKP', 'SKiKS', 'PKIKP', 'SKIKS']
DEFAULT_SERVICE_VERSIONS = {"timeseries": 1, "sacpz": 1, "resp": 1,
                            "evalresp": 1, "traveltime": 1, "flinnengdahl": 2,
                            "distaz": 1}


class Client(object):
    """
    EarthScope Web service request client.

    :type base_url: str, optional
    :param base_url: Base URL of the EarthScope Web service (default
        is ``'https://service.earthscope.org/irisws'``).
    :type user: str, optional
    :param user: The user name used for authentication with the Web
        service (default an empty string).
    :type password: str, optional
    :param password: A password used for authentication with the Web
        service (default is an empty string).
    :type timeout: int, optional
    :param timeout: Seconds before a connection timeout is raised (default
        is ``10`` seconds).
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

    >>> from obspy.clients.iris import Client
    >>> client = Client()
    >>> dt = UTCDateTime("2005-01-01")
    >>> data = client.evalresp("IU", "ANMO", "00", "BHZ", dt, output='fap')
    >>> data[0]  # frequency, amplitude, phase of first point
    array([  1.00000000e-05,   1.05593400e+04,   1.79200700e+02])
    """
    def __init__(self, base_url="https://service.earthscope.org/irisws",
                 user="", password="", timeout=20, debug=False,
                 user_agent=DEFAULT_USER_AGENT, major_versions={}):
        """
        Initializes the EarthScope Web service client.

        See :mod:`obspy.clients.iris` for all parameters.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.debug = debug
        self.user_agent = user_agent
        self.major_versions = DEFAULT_SERVICE_VERSIONS
        self.major_versions.update(major_versions)
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib_request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib_request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib_request.build_opener(auth_handler)
        # install globally
        urllib_request.install_opener(opener)

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
            options += urlencode(params)
        if options:
            remoteaddr = "%s?%s" % (remoteaddr, options)
        if self.debug:
            print('\nRequesting %s' % (remoteaddr))
        req = urllib_request.Request(url=remoteaddr, data=data,
                                     headers=headers)
        response = urllib_request.urlopen(req, timeout=self.timeout)
        doc = response.read()
        return doc

    def _to_file_or_data(self, filename, data, binary=False):
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
        elif isinstance(filename, str):
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

    # new deprecation in 1.5.1, remove in 1.6.0 or 1.7.0
    @deprecated()
    def timeseries(self, network, station, location, channel,
                   starttime, endtime, filter=[], filename=None,
                   output='miniseed', **kwargs):
        """
        DEPRECATED as of 1.5.1 - will be removed in future release

        EarthScope has announced the retirement of its "irisws-timeseries" web
        service for August 26, 2026. For details see
        https://www.earthscope.org/news/mailing-lists/. This method will be
        removed in a future obspy release, so please adjust accordingly.
        """
        kwargs['network'] = str(network)
        kwargs['station'] = str(station)
        if location:
            kwargs['location'] = str(location)[0:2]
        else:
            kwargs['location'] = '--'
        kwargs['channel'] = str(channel)
        # convert UTCDateTime to string for query
        kwargs['starttime'] = UTCDateTime(starttime).format_iris_web_service()
        kwargs['endtime'] = UTCDateTime(endtime).format_iris_web_service()
        # output
        if filename:
            kwargs['output'] = output
        else:
            kwargs['output'] = 'miniseed'
        # build up query
        try:
            data = self._fetch("timeseries", param_list=filter, **kwargs)
        except urllib_request.HTTPError as e:
            msg = "No waveform data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        # write directly if file name is given
        if filename:
            return self._to_file_or_data(filename, data, True)
        # create temporary file for writing data
        with NamedTemporaryFile() as tf:
            tf.write(data)
            # read stream using obspy.io.mseed
            tf.seek(0)
            try:
                stream = read(tf.name, 'MSEED')
            except Exception:
                stream = Stream()
        return stream

    # new deprecation in 1.5.1, remove in 1.6.0 or 1.7.0
    @deprecated()
    def resp(self, network, station, location="*", channel="*",
             starttime=None, endtime=None, filename=None, **kwargs):
        """
        DEPRECATED as of 1.5.1 - will be removed in future release

        EarthScope has announced the retirement of its "irisws-resp" web
        service for August 31, 2026. For details see
        https://www.earthscope.org/news/mailing-lists/. This method will
        be removed in a future obspy release, so please adjust accordingly.
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
                    UTCDateTime(starttime).format_iris_web_service()
            except Exception:
                kwargs['starttime'] = starttime
            try:
                kwargs['endtime'] = \
                    UTCDateTime(endtime).format_iris_web_service()
            except Exception:
                kwargs['endtime'] = endtime
        elif 'time' in kwargs:
            try:
                kwargs['time'] = \
                    UTCDateTime(kwargs['time']).format_iris_web_service()
            except Exception:
                pass
        # build up query
        try:
            data = self._fetch("resp", **kwargs)
        except urllib_request.HTTPError as e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return self._to_file_or_data(filename, data)

    # new deprecation in 1.5.1, remove in 1.6.0 or 1.7.0
    @deprecated()
    def sacpz(self, network, station, location="*", channel="*",
              starttime=None, endtime=None, filename=None, **kwargs):
        """
        DEPRECATED as of 1.5.1 - will be removed in future release

        EarthScope has announced the retirement of its "irisws-sacpz" web
        service for August 31, 2026. For details see
        https://www.earthscope.org/news/mailing-lists/. This method will
        be removed in a future obspy release, so please adjust accordingly.
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
                    UTCDateTime(starttime).format_iris_web_service()
            except Exception:
                kwargs['starttime'] = starttime
            try:
                kwargs['endtime'] = \
                    UTCDateTime(endtime).format_iris_web_service()
            except Exception:
                kwargs['endtime'] = endtime
        elif starttime:
            try:
                kwargs['time'] = \
                    UTCDateTime(starttime).format_iris_web_service()
            except Exception:
                kwargs['time'] = starttime
        data = self._fetch("sacpz", **kwargs)
        return self._to_file_or_data(filename, data)

    # new deprecation in 1.5.1, remove in 1.6.0 or 1.7.0
    @deprecated()
    def distaz(self, stalat, stalon, evtlat, evtlon):
        """
        DEPRECATED as of 1.5.1 - will be removed in future release

        EarthScope has announced the retirement of its "irisws-distaz"
        web service for August 27, 2026. For details see
        https://www.earthscope.org/news/mailing-lists/. This method will
        be removed in a future obspy release, so please adjust accordingly.
        """
        # build up query
        try:
            data = self._fetch("distaz", stalat=stalat, stalon=stalon,
                               evtlat=evtlat, evtlon=evtlon)
        except urllib_request.HTTPError as e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        data = objectify.fromstring(data.decode())
        results = {}
        results['ellipsoidname'] = str(data.ellipsoid.attrib['name'])
        results['distance'] = float(data.distance)
        results['distancemeters'] = float(data.distanceMeters)
        results['backazimuth'] = float(data.backAzimuth)
        results['azimuth'] = float(data.azimuth)
        return results

    @deprecated()
    def flinnengdahl(self, lat, lon, rtype="both"):
        """
        DEPRECATED as of 1.5.1 - will be removed in future release

        EarthScope has announced the retirement of its Flinn-Engdahl web
        service on or after July 6th 2026. Try using
        obspy.geodetics.flinnengdahl instead.
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
        except urllib_request.HTTPError as e:
            msg = "No Flinn-Engdahl data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)

    # new deprecation in 1.5.1, remove in 1.6.0 or 1.7.0
    @deprecated()
    def traveltime(self, model='iasp91', phases=DEFAULT_PHASES, evdepth=0.0,
                   distdeg=None, distkm=None, evloc=None, staloc=None,
                   noheader=False, traveltimeonly=False, rayparamonly=False,
                   mintimeonly=False, filename=None):
        """
        DEPRECATED as of 1.5.1 - will be removed in future release

        EarthScope has announced the retirement of its "irisws-traveltime"
        web service for August 27, 2026. For details see
        https://www.earthscope.org/news/mailing-lists/. This method will
        be removed in a future obspy release, so please adjust accordingly.
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
        except urllib_request.HTTPError as e:
            msg = "No response data available (%s: %s)"
            msg = msg % (e.__class__.__name__, e)
            raise Exception(msg)
        return self._to_file_or_data(filename, data)

    def evalresp(self, network, station, location, channel, time=UTCDateTime(),
                 minfreq=0.00001, maxfreq=None, nfreq=200, units='def',
                 width=800, height=600, annotate=True, output='plot',
                 filename=None, **kwargs):
        """
        Low-level interface for `evalresp` Web service of EarthScope
        (https://service.earthscope.org/irisws/evalresp/) - release 1.0.0
        (2011-08-11).

        This method evaluates instrument response information stored at the
        EarthScope DMC and outputs ASCII data or
        `Bode Plots <https://en.wikipedia.org/wiki/Bode_plots>`_.

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

            >>> from obspy.clients.iris import Client
            >>> client = Client()
            >>> dt = UTCDateTime("2005-01-01")
            >>> data = client.evalresp("IU", "ANMO", "00", "BHZ", dt,
            ...                        output='fap')
            >>> data[0]  # frequency, amplitude, phase of first point
            array([  1.00000000e-05,   1.05593400e+04,   1.79200700e+02])

        (2) Returning amplitude and phase plot.

            >>> from obspy.clients.iris import Client
            >>> client = Client()
            >>> dt = UTCDateTime("2005-01-01")
            >>> client.evalresp("IU", "ANMO", "00", "BHZ", dt) # doctest: +SKIP

            .. plot::

                from obspy import UTCDateTime
                from obspy.clients.iris import Client
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
            kwargs['time'] = UTCDateTime(time).format_iris_web_service()
        except Exception:
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
                    img = image._png.read_png(tf.name)
                # add image to axis
                ax.imshow(img)
                # hide axes
                ax.axison = False
                # show plot
                plt.show()
            else:
                self._to_file_or_data(filename, data, binary=True)
        else:
            # ASCII data
            if filename is None:
                return np.loadtxt(io.BytesIO(data), ndmin=1)
            else:
                return self._to_file_or_data(filename, data, binary=True)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
