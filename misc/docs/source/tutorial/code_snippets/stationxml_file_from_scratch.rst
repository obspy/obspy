=============================================================================
Creating a StationXML file from Scratch
=============================================================================

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
:class:`~obspy.core.inventory.channel.Channel` objects. Instrument
:class:`~obspy.core.inventory.response.Response` objects are part of the
channels and not discussed here.


.. code-block:: python

    from obspy.core.inventory import Inventory, Network, Station, Channel, Site
    import obspy


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
        code="EHZ",
        # This is the location code according to the SEED standard.
        location_code="",
        # Note that these coordinates can differ from the station coordinates.
        latitude=1.0,
        longitude=2.0,
        elevation=345.0,
        depth=10.0,
        azimuth=0.0,
        dip=-90.0)


    # Now tie it all together.
    inv.networks.append(net)
    net.stations.append(sta)
    sta.channels.append(cha)

    # And finally write it to a StationXML file. We also force a validation against
    # the StationXML schema to ensure it produces a valid StationXML file.
    #
    # Note that it is also possible to serialize to any of the other inventory
    # output formats ObsPy supports.
    inv.write("station.xml", format="stationxml", validate=True)
