# -*- coding: utf-8 -*-
"""
A simplified interface to the obspy.seedlink module.

The :class:`~obspy.seedlink.easyseedlink.EasySeedLinkClient` class contained in
this module provides a more pythonic interface to the :mod:`obspy.seedlink`
module with a focus on ease of use, while minimizing unnecessary exposure of
the protocol specifics.

A client object can easily be created using the
:func:`~obspy.seedlink.easyseedlink.create_client` function, e.g. by providing
a function to handle incoming data and a server URL:

.. code-block:: python

    from obspy.seedlink.easyseedlink import create_client

    # A function to handle incoming data
    def handle_data(trace):
        print('Received the following trace:')
        print(trace)
        print()

    # Create the client and pass the function as a callback
    client = create_client('geofon.gfz-potsdam.de', on_data=handle_data)
    client.select_stream('BW', 'MANZ', 'EHZ')
    client.run()

For advanced applications, subclassing the
:class:`~obspy.seedlink.easyseedlink.EasySeedLinkClient` class allows for more
flexibility. See the
:class:`~obspy.seedlink.easyseedlink.EasySeedLinkClient` documentation
for an example.

.. note::

    For finer grained control of the SeedLink connection (e.g. custom
    processing of individual SeedLink packets), using
    :class:`~obspy.seedlink.client.seedlinkconnection.SeedLinkConnection` or
    :class:`~obspy.seedlink.slclient.SLClient` directly might be the preferred
    option.

.. rubric:: Limitations

As of now, single station mode is not supported. Neither are in-stream ``INFO``
requests.

The client is using the
:class:`~obspy.seedlink.client.seedlinkconnection.SeedLinkConnection` class and
hence inherits all of its limitations. For example, erroneous packets are only
logged, but otherwise ignored, with no possibility of handling them
explicitly. Keepalive handling is completely encapsulated inside the
connection object and cannot be easily influenced. Also, a ``HELLO`` is always
sent to the server when connecting in order to determine the SeedLink protocol
version.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import urlparse
import lxml

from obspy.seedlink.client.seedlinkconnection import SeedLinkConnection
from obspy.seedlink.slpacket import SLPacket
from obspy.seedlink.client.slstate import SLState


class EasySeedLinkClientException(Exception):
    """
    A base exception for all errors triggered explicitly by EasySeedLinkClient.
    """
    # XXX Base on SeedLinkException?
    pass


class EasySeedLinkClient(object):
    """
    An easy-to-use SeedLink client.

    This class is meant to be used as a base class, with a subclass
    implementing one or more of the callbacks (most usefully the
    :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.on_data` callback).
    See the :ref:`ObsPy Tutorial <seedlink-tutorial>` for a detailed example.

    .. rubric:: Example

    .. code-block:: python

        # Subclass the client class
        class MyClient(EasySeedLinkClient):
            # Implement the on_data callback
            def on_data(self, trace):
                print('Received trace:')
                print(trace)

        # Connect to a SeedLink server
        client = MyClient('geofon.gfz-potsdam.de:18000')

        # Retrieve INFO:STREAMS
        streams_xml = client.get_info('STREAMS')
        print(streams_xml)

        # Select a stream and start receiving data
        client.select_stream('BW', 'RJOB', 'EHZ')
        client.run()

    .. rubric:: Implementation

    The EasySeedLinkClient uses the
    :class:`~obspy.seedlink.client.seedlinkconnection.SeedLinkConnection`
    object. (It is not based on
    :class:`~obspy.seedlink.slclient.SLClient`.)

    :type server_url: str
    :param server_url: The SeedLink server URL
    :type autoconnect: bool
    :param autoconnect: Connect to the server when the client object is
                        created; default is True.

    .. warning::

        The SeedLink connection only fails on connection errors if the
        connection was started explicitly, either when ``autoconnect`` is
        ``True`` or by calling
        :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.connect`
        explicitly. Otherwise the client might get stuck in an infinite
        reconnect loop if there are connection problems (e.g. connect, timeout,
        reconnect, timeout, ...). This might be intended behavior in some
        situations.
    """

    def __init__(self, server_url, autoconnect=True):
        # Catch invalid server_url parameters
        if not isinstance(server_url, basestring):
            raise ValueError('Expected string for SeedLink server URL')
        # Allow for sloppy server URLs (e.g. 'geofon.gfz-potsdam.de:18000).
        # (According to RFC 1808 the net_path segment needs to start with '//'
        # and this is expected by the urlparse function, so it is silently
        # added if it was omitted by the user.)
        if '://' not in server_url and not server_url.startswith('//'):
            server_url = '//' + server_url

        parsed_url = urlparse.urlparse(server_url, scheme='seedlink')

        # Check the provided scheme
        if not parsed_url.scheme == 'seedlink':
            msg = 'Unsupported scheme %s (expected "seedlink")' % \
                  parsed_url.scheme
            raise EasySeedLinkClientException(msg)
        if not parsed_url.hostname:
            msg = 'No host name provided'
            raise EasySeedLinkClientException(msg)

        self.server_hostname = parsed_url.hostname
        self.server_port = parsed_url.port or 18000

        self.conn = SeedLinkConnection()
        self.conn.setSLAddress('%s:%d' %
                               (self.server_hostname, self.server_port))

        if autoconnect:
            self.connect()

        # A flag to indicate if the client has entered streaming mode
        self.__streaming_started = False

        self.__capabilities = None

    def connect(self):
        """
        Connect to the SeedLink server.
        """
        # XXX Check if already connected?
        self.conn.connect()
        self.conn.state.state = SLState.SL_UP

    def get_info(self, level):
        """
        Send a SeedLink ``INFO`` command and retrieve response.

        Available info levels depend on the server implementation. Usually one
        of ``ID``, ``CAPABILITIES``, ``STATIONS``, ``STREAMS``, ``GAPS``,
        ``CONNNECTIONS``, ``ALL``.

        As a convenience, the server's ``CAPABILITIES`` can be accessed through
        the client's
        :attr:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.capabilities`
        attribute.

        .. note::

            This is a synchronous call. While the client waits for the
            response, other packets the server might potentially send will
            be disregarded.

        :type level: str
        :param level: The INFO level to retrieve (sent as ``INFO:LEVEL``)
        """
        if self.__streaming_started:
            msg = 'Method not available after SeedLink connection has ' + \
                  'entered streaming mode.'
            raise EasySeedLinkClientException(msg)

        # Send the INFO request
        self.conn.requestInfo(level)

        # Wait for full response
        while True:
            data = self.conn.collect()

            if data == SLPacket.SLTERMINATE:
                msg = 'SeedLink connection terminated while expecting ' + \
                      'INFO response'
                raise EasySeedLinkClientException(msg)
            elif data == SLPacket.SLERROR:
                msg = 'Unknown error occured while expecting INFO response'
                raise EasySeedLinkClientException(msg)

            # Wait for the terminated INFO response
            packet_type = data.getType()
            if packet_type == SLPacket.TYPE_SLINFT:
                return self.conn.getInfoString()

    @property
    def capabilities(self):
        """
        The server's capabilities, parsed from ``INFO:CAPABILITIES`` (cached).
        """
        if self.__capabilities is None:
            self.__capabilities = []

            capabilities_xml = self.get_info('CAPABILITIES')

            # The INFO response should be encoded in UTF-8. However, if the
            # encoding is given in the XML header (e.g. by IRIS Ringserver),
            # lxml accepts byte input only (and raises a ValueError otherwise.)
            #
            # Example XML header with encoding:
            #     <?xml version="1.0" encoding="utf-8"?>
            try:
                root = lxml.etree.fromstring(capabilities_xml)
            except ValueError:
                root = lxml.etree.fromstring(capabilities_xml.encode('UTF-8'))

            nodes = root.findall('capability')

            for node in nodes:
                self.__capabilities.append(node.attrib['name'].lower())

        return self.__capabilities

    def has_capability(self, capability):
        """
        Check if the SeedLink server has a certain capability.

        The capabilities are fetched using an ``INFO:CAPABILITIES`` request.

        :type capability: str
        :param capability: The capability to check for

        :rtype: bool
        :return: Whether the server has the given capability
        """
        return capability.lower() in self.capabilities

    def has_info_capability(self, capability):
        """
        A shortcut for checking for ``INFO`` capabilities.

        Calling this is equivalent to calling
        :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.has_capability`
        with ``'info:' + capability``.

        .. rubric:: Example

        .. code-block:: python

            # Check if the server has the INFO:STREAMS capability
            client.has_info_capability('STREAMS')

        :type capability: str
        :param capability: The ``INFO`` capability to check for

        :rtype: bool
        :return: Whether the server has the given ``INFO`` capability
        """
        return self.has_capability('info:' + capability)

    def _send_and_recv(self, bytes_, stop_on=[b'END']):
        """
        Send a command to the server and read the response.

        The response is read until a packet is received that ends with one of
        the provided stop words.

        .. warning::

            If the server doesn't send one of the stop words, this never
            returns!

        :type bytes_: str (Python 2) or bytes (Python 3)
        :param bytes_: The bytes to send to the server
        :type stop_on: list
        :param stop_on: A list of strings that indicate the end of the server
                        response.

        :rtype: str (Python 2) or bytes (Python 3)
        :return: The server's response
        """
        if not bytes_.endswith(b'\r'):
            bytes_ += b"\r"
        if not type(stop_on) is list:
            stop_on = [stop_on]
        for i, stopword in enumerate(stop_on):
            if not type(stopword) == bytes:
                stop_on[i] = stopword.encode()

        self.conn.socket.send(bytes_)

        response = bytearray()
        while True:
            bytes_read = self.conn.socket.recv(
                SeedLinkConnection.DFT_READBUF_SIZE)
            response += bytes_read
            for stopword in stop_on:
                if response.endswith(stopword):
                    # Collapse the bytearray
                    return bytes(response)

    def _get_CAT(self):
        """
        Send the CAT command to a server and receive the answer.

        This can potentially be used for older SeedLink servers that don't
        support the ``INFO:STREAMS`` command yet.
        """
        # Quick hack, but works so far
        ringserver_error = 'CAT command not implemented\r\n'

        response = self._send_and_recv('CAT', ['END', ringserver_error])

        if response == ringserver_error:
            raise EasySeedLinkClientException(ringserver_error.strip())

        return response

    def select_stream(self, net, station, selector=None):
        """
        Select a stream for data transfer.

        This method can be called once or multiple times as needed. A
        subsequent call to the
        :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.run` method
        starts the streaming process.

        .. note::
            Selecting a stream always puts the SeedLink connection in
            *multi-station mode*, even if only a single stream is selected.
            *Uni-station mode* is not supported.

        :type net: str
        :param net: The network id
        :type station: str
        :param station: The station id
        :type selectors: str
        :param selector: a valid SeedLink selector, e.g. ``EHZ`` or ``EH?``
        """
        if not self.has_capability('multistation'):
            msg = 'SeedLink server does not support multi-station mode'
            raise EasySeedLinkClientException(msg)

        if self.__streaming_started:
            msg = 'Adding streams is not supported after the SeedLink ' + \
                  'connection has entered streaming mode.'
            raise EasySeedLinkClientException(msg)

        self.conn.addStream(net, station, selector, seqnum=-1, timestamp=None)

    def run(self):
        """
        Start streaming data from the SeedLink server.

        Streams need to be selected using
        :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.select_stream`
        before this is called.

        This method enters an infinite loop, calling the client's callbacks
        when events occur.
        """
        # Note: This somewhat resembles the run() method in SLClient.

        # Check if any streams have been specified (otherwise this will result
        # in an infinite reconnect loop in the SeedLinkConnection)
        if not len(self.conn.streams):
            msg = 'No streams specified. Use select_stream() to select ' + \
                  'a stream.'
            raise EasySeedLinkClientException(msg)

        self.__streaming_started = True

        # Start the collection loop
        while True:
            data = self.conn.collect()

            if data == SLPacket.SLTERMINATE:
                self.on_terminate()
                break
            elif data == SLPacket.SLERROR:
                self.on_seedlink_error()
                continue

            # At this point the received data should be a SeedLink packet
            # XXX In SLClient there is a check for data == None, but I think
            #     there is no way that self.conn.collect() can ever return None
            assert(isinstance(data, SLPacket))

            packet_type = data.getType()

            # Ignore in-stream INFO packets (not supported)
            if packet_type not in (SLPacket.TYPE_SLINF, SLPacket.TYPE_SLINFT):
                # The packet should be a data packet
                trace = data.getTrace()
                # Pass the trace to the on_data callback
                self.on_data(trace)

    def close(self):
        """
        Close the SeedLink connection.

        .. note::

            Closing  the connection is not threadsafe yet. Client code must
            ensure that
            :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.run` and
            :meth:`SeedLinkConnection.terminate()
            <obspy.seedlink.client.seedlinkconnection.SeedLinkConnection.terminate>`
            are not being called after the connection has been closed.

            See the corresponding `GitHub issue
            <https://github.com/obspy/obspy/pull/876#issuecomment-60537414>`_
            for details.
        """
        self.conn.disconnect()

    def on_terminate(self):
        """
        Callback for handling connection termination.

        A termination event can either be triggered by the SeedLink server
        explicitly terminating the connection (by sending an ``END`` packet in
        streaming mode) or by the
        :meth:`~obspy.seedlink.client.seedlinkconnection.SeedLinkConnection.terminate`
        method of the
        :class:`~obspy.seedlink.client.seedlinkconnection.SeedLinkConnection`
        object being called.
        """
        pass

    def on_seedlink_error(self):
        """
        Callback for handling SeedLink errors.

        This handler is called when an ``ERROR`` response is received. The
        error generally corresponds to the last command that was sent to the
        server. However, with the current implementation of the SeedLink
        connection, no further information about the error is available.
        """
        pass

    def on_data(self, trace):
        """
        Callback for handling the reception of waveform data.

        Override this for data streaming.

        :type trace: :class:`~obspy.core.trace.Trace`
        :param trace: The trace received from the server
        """
        pass


def create_client(server_url, on_data=None, on_seedlink_error=None,
                  on_terminate=None):
    """
    Quickly create an EasySeedLinkClient instance.

    .. rubric:: Example

    .. code-block:: python

        >>> from obspy.seedlink.easyseedlink import create_client

        >>> def handle_data(trace):
        ...     print('Received new data:')
        ...     print(trace)
        ...     print()
        ...
        >>> client = create_client('geofon.gfz-potsdam.de', handle_data)
        >>> client.select_stream('BW', 'MANZ', 'EHZ')
        >>> client.run()

    .. note::

        The methods passed to the :func:`create_client` function are not bound
        to the client instance, i.e. they do not have access to the instance
        via the ``self`` attribute. To get a bound method, the client class
        can be subclassed and the method overridden.

    :type server_url: str
    :param server_url: The SeedLink server URL
    :type on_data: function or callable
    :param on_data: A function or callable that is called for every new trace
                    received from the server; needs to accept one argument (the
                    trace); default is ``None``
    :type on_seedlink_error: function or callable
    :param on_seedlink_error: A function or callable that is called when a
                              SeedLink ERROR response is received (see the
                              :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.on_seedlink_error`
                              method for details); default is ``None``
    :type on_terminate: function or callable
    :param on_terminate: A function or callable that is called when the
                         connection is terminated (see the
                         :meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.on_terminate`
                         method for details); default is ``None``
    """
    client = EasySeedLinkClient(server_url, autoconnect=False)

    not_callable_error = 'A callable must be passed to %s'

    if on_data is not None:
        if not callable(on_data):
            raise EasySeedLinkClientException(not_callable_error % 'on_data')
        client.on_data = on_data

    if on_seedlink_error is not None:
        if not callable(on_seedlink_error):
            raise EasySeedLinkClientException(not_callable_error %
                                              'on_seedlink_error')
        client.on_seedlink_error = on_seedlink_error

    if on_terminate is not None:
        if not callable(on_terminate):
            raise EasySeedLinkClientException(not_callable_error %
                                              'on_terminate')
        client.on_terminate = on_terminate

    client.connect()

    return client
