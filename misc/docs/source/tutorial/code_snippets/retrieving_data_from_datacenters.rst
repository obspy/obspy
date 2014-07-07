=================================
Retrieving Data from Data Centers
=================================

----------------
ArcLink Protocol
----------------

In the following example uses the :mod:`obspy.arclink` module in order to
retrieve waveforms data as well as poles and zeros from a remote server
via the `ArcLink <http://www.seiscomp3.org/wiki/doc/applications/arclink>`_
protocol. The retrieved poles and zeros are then used to correct for the
instrument response and to simulate a 1Hz instrument with damping 0.707.

.. note::
    The default client needs to open port 18002 to the host webdc.eu via TCP/IP
    in order to download the requested data. Please make sure that no firewall
    is blocking access to this server/port combination.

.. note::
    The ``user`` keyword in the following example is used for identification
    with the ArcLink server as well as for usage statistics within the data
    center, so please provide a meaningful user id such as your email address.

.. plot:: tutorial/code_snippets/retrieving_data_from_datacenters_1.py
   :include-source:

-----------------
IRIS Web Services
-----------------

TODO - for now see :mod:`obspy.iris`

---------------------
Earthworm Wave Server
---------------------

TODO - for now see :mod:`obspy.earthworm`

-------------------
NERIES Web Services
-------------------

TODO - for now see :mod:`obspy.neries`
