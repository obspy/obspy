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
    :mod:`obspy.fdsn` module for this. It supports the largest number of data
    centers and uses the most modern data formats. There are still a number of
    reasons to choose a different module but please make sure you have one.

---------------------
The FDSN Web Services
---------------------

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
most likely want to use the** :mod:`obspy.fdsn` **module**. It is able to
request data from any data center implementing the `FDSN web services
<http://www.fdsn.org/webservices/>`_. Example data centers include
IRIS/ORFEUS/INGV/ETH/GFZ/RESIF/... - a curated list can be found `here
<http://www.fdsn.org/webservices/datacenters/>`_. As a further advantage it
returns data in the most modern and future proof formats.

-------
ArcLink
-------

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | MiniSEED, SEED                 |
+----------------------+--------------------------------+
| Station Information  | dataless SEED, SEED            |
+----------------------+--------------------------------+

ArcLink is a distributed data request protocol usable to access archived
waveform data in the MiniSEED or SEED format and associated meta information as
Dataless SEED files. You can use the :mod:`obspy.arclink` module to request
data from the `EIDA <http://www.orfeus-eu.org/eida/>`_ initiative but most (or
all) of that data can also be requested using the :mod:`obspy.fdsn` module.

-----------------
IRIS Web Services
-----------------

**Available Data Types and Formats:** Various

IRIS (in addition to FDSN web services) offers a variety of special-purpose web
services, for some of which ObsPy has interfaces in the :mod:`obspy.iris`
module. Use this if you require response information in the SAC poles & zeros
or in the RESP format. If you just care about the instrument response, please
use the :mod:`obspy.fdsn` module to request StationXML data which contains the
same information.

The interfaces for the calculation tools are kept around for legacy reasons;
internal ObsPy functionality should be considered as an alternative when
working within ObsPy:

+--------------------------------------------------+--------------------------------------------------------------+
| IRIS Web Service                                 | Equivalent ObsPy Function/Module                             |
+==================================================+==============================================================+
| :meth:`obspy.iris.client.Client.traveltime()`    | :mod:`obspy.taup`                                            |
+--------------------------------------------------+--------------------------------------------------------------+
| :meth:`obspy.iris.client.Client.distaz()`        | :mod:`obspy.core.util.geodetics`                             |
+--------------------------------------------------+--------------------------------------------------------------+
| :meth:`obspy.iris.client.Client.flinnengdahl()`  | :class:`obspy.core.util.geodetics.flinnengdahl.FlinnEngdahl` |
+--------------------------------------------------+--------------------------------------------------------------+

---------------------
Earthworm Wave Server
---------------------

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | Custom Format                  |
+----------------------+--------------------------------+

Use the :mod:`obspy.earthworm` module to request data from the `Earthworm
<http://www.earthwormcentral.org/>`_ data acquisition system.

-------------------
NERIES Web Services
-------------------

This service is largely deprecated as the data can just as well be requested
via the :mod:`obspy.fdsn` module.

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
:mod:`obspy.neic` module to request data from it.

--------
SeedLink
--------

+----------------------+--------------------------------+
| Available Data Types | Format                         |
+======================+================================+
| Waveforms            | MiniSEED                       |
+----------------------+--------------------------------+

To connect to a real time SeedLink server, use the :mod:`obspy.seedlink`
module. Also see the :ref:`ObsPy Tutorial <seedlink-tutorial>` for a more
detailed introduction.
