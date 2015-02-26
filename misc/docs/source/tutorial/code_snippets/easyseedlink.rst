.. _seedlink-tutorial:

===============================
Connecting to a SeedLink Server
===============================

The :mod:`obspy.seedlink` module provides a Python implementation of the
SeedLink client protocol. The :mod:`obspy.seedlink.easyseedlink` submodule
contains a high-level interface to the SeedLink implementation that facilitates
the creation of a SeedLink client.

--------------------------
The create_client function
--------------------------

The easiest way to connect to a SeedLink server is using the
:func:`~obspy.seedlink.easyseedlink.create_client` function to create a new
instance of the :class:`~obspy.seedlink.easyseedlink.EasySeedLinkClient` class.
It accepts as an argument a function that handles new data received from the
SeedLink server, for example:

.. code-block:: python

    def handle_data(trace):
        print('Received the following trace:')
        print(trace)
        print()

This function can then be passed to
:func:`~obspy.seedlink.easyseedlink.create_client` together with a SeedLink
server URL to create a client instance:

.. code-block:: python

    client = create_client('geofon.gfz-potsdam.de', on_data=handle_data)

The client immediately connects to the server when it is created.

Sending INFO requests to the server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The client instance can be used to send SeedLink ``INFO`` requests to the
server:

.. code-block:: python

    # Send the INFO:ID request
    client.get_info('ID')

    # Returns:
    # <?xml version="1.0"?>\n<seedlink software="SeedLink v3.2 (2014.071)" organization="GEOFON" started="2014/09/01 14:08:37.4192"/>\n

The responses to ``INFO`` requests are in XML format. The client provides a
shortcut to retrieve and parse the server's capabilities (via an
``INFO:CAPABILITIES`` request):

.. code-block:: python

    >>> client.capabilities
    ['dialup', 'multistation', 'window-extraction', 'info:id', 'info:capabilities', 'info:stations', 'info:streams']

The capabilities are fetched and parsed when the attribute is first accessed
and are cached after that.

Streaming data from the server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to start receiving waveform data, a *stream* needs to be selected.
This is done by calling the
:meth:`~obspy.seedlink.easyseedlink.EasySeedLinkClient.select_stream` method:

.. code-block:: python

    client.select_stream('BW', 'MANZ', 'EHZ')

Multiple streams can be selected. SeedLink wildcards are also supported:

.. code-block:: python

    client.select_stream('BW', 'ROTZ', 'EH?')

After having selected the streams, the client is ready to enter streaming mode:

.. code-block:: python

    client.run()

This starts streaming data from the server. Upon every complete trace that is
received from the server, the function defined above is called with the trace
object:

.. code-block:: python

	Received new data:
	BW.MANZ..EHZ | 2014-09-04T19:47:25.625000Z - 2014-09-04T19:47:26.770000Z | 200.0 Hz, 230 samples

	Received new data:
	BW.ROTZ..EHZ | 2014-09-04T19:47:22.685000Z - 2014-09-04T19:47:24.740000Z | 200.0 Hz, 412 samples

	Received new data:
	BW.ROTZ..EHZ | 2014-09-04T19:47:24.745000Z - 2014-09-04T19:47:26.800000Z | 200.0 Hz, 412 samples

	Received new data:
	BW.ROTZ..EHN | 2014-09-04T19:47:20.870000Z - 2014-09-04T19:47:22.925000Z | 200.0 Hz, 412 samples

	Received new data:
	BW.ROTZ..EHN | 2014-09-04T19:47:22.930000Z - 2014-09-04T19:47:24.985000Z | 200.0 Hz, 412 samples

The :func:`~obspy.seedlink.easyseedlink.create_client` function also accepts
functions to be called when the connection terminates or when a SeedLink
error is received.
See the :func:`documentation <obspy.seedlink.easyseedlink.create_client>` for
details.

--------------------------------------
Advanced usage: subclassing the client
--------------------------------------

For advanced use cases, subclassing the
:class:`~obspy.seedlink.easyseedlink.EasySeedLinkClient` class allows for finer
control over the instance. Implementing the same client as above:

.. code-block:: python

    class DemoClient(EasySeedLinkClient):
        """
        A custom SeedLink client
        """
        def on_data(self, trace):
            """
            Override the on_data callback
            """
            print('Received trace:')
            print(trace)
            print()

The :class:`documentation <obspy.seedlink.easyseedlink.EasySeedLinkClient>`
has more details about the client.
