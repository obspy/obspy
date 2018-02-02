=======================================
Creating a StationXML file from Scratch
=======================================

Creating a custom StationXML file is a task that sometimes comes up in
seismology. This section demonstrates how to it with ObsPy. Please note that
this is not necessarily easier or more obvious then directly editing an XML
file but it does provider tighter integration with the rest of ObsPy and can
guarantee a valid result at the end.

Note that this assumes a certain familiarity with the `FDSN StationXML standard
<https://www.fdsn.org/xml/station/>`_. We'll create a fairly simplistic
StationXML file and many arguments are optional. ObsPy will validate the
resulting StationXML file against its schema upon writing so the final file is
assured to be valid against the StationXML schema.

The following illustration shows the basic structure of ObsPy's internal
representation.

.. figure:: /_images/Inventory.png

Each big box will be an object and all objects will have to be hierarchically
linked to form a single :class:`~obspy.core.inventory.inventory.Inventory`
object. An inventory can contain any number of
:class:`~obspy.core.inventory.network.Network` objects, which in turn can
contain any number of :class:`~obspy.core.inventory.station.Station` objects,
which once again in turn can contain any number of
:class:`~obspy.core.inventory.channel.Channel` objects. For each channel, the
instrument response  can be stored as the
:class:`response <obspy.core.inventory.response.Response>` attribute.

Instrument Response can be looked up and attached to the channels from the IRIS
DMC `Library of Nominal Responses`_ for Seismic Instruments (NRL) using
ObsPy's :mod:`NRL client <obspy.clients.nrl>`.

.. _Library of Nominal Responses: http://ds.iris.edu/NRL/


.. code-block:: python

    import obspy
    from obspy.core.inventory import Inventory, Network, Station, Channel, Site
    from obspy.clients.nrl import NRL


    # We'll first create all the various objects. These strongly follow the
    # hierarchy of StationXML files.
    inv = Inventory(
        # We'll add networks later.
        networks=[],
        # The source should be the id whoever create the file.
        source="ObsPy-Tutorial")

    net = Network(
        # This is the network code according to the SEED standard.
        code="XX",
        # A list of stations. We'll add one later.
        stations=[],
        description="A test stations.",
        # Start-and end dates are optional.
        start_date=obspy.UTCDateTime(2016, 1, 2))

    sta = Station(
        # This is the station code according to the SEED standard.
        code="ABC",
        latitude=1.0,
        longitude=2.0,
        elevation=345.0,
        creation_date=obspy.UTCDateTime(2016, 1, 2),
        site=Site(name="First station"))

    cha = Channel(
        # This is the channel code according to the SEED standard.
        code="HHZ",
        # This is the location code according to the SEED standard.
        location_code="",
        # Note that these coordinates can differ from the station coordinates.
        latitude=1.0,
        longitude=2.0,
        elevation=345.0,
        depth=10.0,
        azimuth=0.0,
        dip=-90.0,
        sample_rate=200)

    # By default this accesses the NRL online. Offline copies of the NRL can
    # also be used instead
    nrl = NRL()
    # The contents of the NRL can be explored interactively in a Python prompt,
    # see API documentation of NRL submodule:
    # http://docs.obspy.org/packages/obspy.clients.nrl.html
    # Here we assume that the end point of data logger and sensor are already
    # known:
    response = nrl.get_response( # doctest: +SKIP
        sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
        datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])


    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    # And finally write it to a StationXML file. We also force a validation against
    # the StationXML schema to ensure it produces a valid StationXML file.
    #
    # Note that it is also possible to serialize to any of the other inventory
    # output formats ObsPy supports.
    inv.write("station.xml", format="stationxml", validate=True)
