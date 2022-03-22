=================================
Retrieving Data from Data Centers
=================================

This section is intended as a small guide to help you choose the best way to
download data for a given purpose using ObsPy. Keep in mind that data centers
and web services are constantly changing so this recommendation might not be
valid anymore at the time you read this. For actual code examples, please see
the documentation of the various modules.

.. note::

    The most common use case is likely to download waveforms and event/station
    meta information. In almost all cases you will want to use the
    :mod:`obspy.clients.fdsn` module for this. It supports the largest number
    of data centers and uses the most modern data formats. There are still a
    number of reasons to choose a different module but please make sure you
    have one.

---------------------
The FDSN Web Services
---------------------

Basic FDSN Web Services
-----------------------

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | MiniSEED and optionally others |
+----------------------+--------------------------------+
| Station Information  | StationXML and Text            |
+----------------------+--------------------------------+
| Event Information    | QuakeML and Text               |
+----------------------+--------------------------------+

.. note::

    Not all data providers offer all three data types. Many offer only one or two.

If you want to requests waveforms, or station/event meta information **you will
most likely want to use the** :mod:`obspy.clients.fdsn` **module**. It is able
to request data from any data center implementing the `FDSN web services
<https://www.fdsn.org/webservices/>`_. Example data centers include
IRIS/ORFEUS/INGV/ETH/GFZ/RESIF/... - a curated list can be found `here
<https://www.fdsn.org/webservices/datacenters/>`_. As a further advantage it
returns data in the most modern and future proof formats.

FDSN Routing Web Services
-------------------------

If you don't know which data center has what data, use one of the routing
services. ObsPy has support for two of them:

(i) The `IRIS Federator  <https://service.iris.edu/irisws/fedcatalog/1/>`_.
(ii) The `EIDAWS Routing Service
     <http://www.orfeus-eu.org/data/eida/webservices/routing/>`_.

See the bottom part of the :mod:`obspy.clients.fdsn` module page for usage
details.

FDSN Mass Downloader
--------------------

If you want to download a lot of data across a number of data centers,
ObsPy's mass (or batch) downloader is for you. You can formulate your queries
for example in terms of geographical domains and ObsPy will download
waveforms and corresponding station meta information to produce complete
data sets, ready for research, including some basic quality control.

See the :mod:`obspy.clients.fdsn.mass_downloader` page for more details.

-----------------
IRIS Web Services
-----------------

**Available Data Types and Formats:** Various

IRIS (in addition to FDSN web services) offers a variety of special-purpose web
services, for some of which ObsPy has interfaces in the
:mod:`obspy.clients.iris` module. Use this if you require response information
in the SAC poles & zeros or in the RESP format. If you just care about the
instrument response, please use the :mod:`obspy.clients.fdsn` module to request
StationXML data which contains the same information.

The interfaces for the calculation tools are kept around for legacy reasons;
internal ObsPy functionality should be considered as an alternative when
working within ObsPy:

+---------------------------------------------------------+--------------------------------------------------------------+
| IRIS Web Service                                        | Equivalent ObsPy Function/Module                             |
+=========================================================+==============================================================+
| :meth:`obspy.clients.iris.client.Client.traveltime()`   | :mod:`obspy.taup`                                            |
+---------------------------------------------------------+--------------------------------------------------------------+
| :meth:`obspy.clients.iris.client.Client.distaz()`       | :mod:`obspy.geodetics`                                       |
+---------------------------------------------------------+--------------------------------------------------------------+
| :meth:`obspy.clients.iris.client.Client.flinnengdahl()` | :class:`obspy.geodetics.flinnengdahl.FlinnEngdahl`           |
+---------------------------------------------------------+--------------------------------------------------------------+

---------------------
Earthworm Wave Server
---------------------

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | Custom Format                  |
+----------------------+--------------------------------+

Use the :mod:`obspy.clients.earthworm` module to request data from the
`Earthworm <http://www.earthwormcentral.org/>`_ data acquisition system.

-------------------
NERIES Web Services
-------------------

This service is largely deprecated as the data can just as well be requested
via the :mod:`obspy.clients.fdsn` module.

----
NEIC
----

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | MiniSEED                       |
+----------------------+--------------------------------+

The Continuous Waveform Buffer (CWB) is a repository for seismic waveform data
that passes through the NEIC “Edge” processing system. Use the
:mod:`obspy.clients.neic` module to request data from it.

--------
SeedLink
--------

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | MiniSEED                       |
+----------------------+--------------------------------+

To connect to a real time SeedLink server, use the
:mod:`obspy.clients.seedlink` module. Also see the
:ref:`ObsPy Tutorial <seedlink-tutorial>` for a more detailed introduction.

---------------
Syngine Service
---------------

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | MiniSEED and zipped SAC files  |
+----------------------+--------------------------------+

Use the :mod:`obspy.clients.syngine` module to download high-frequency global
synthetic seismograms for any source receiver combination from the IRIS syngine
service.
