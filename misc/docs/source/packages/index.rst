============
API Overview
============

ObsPy's functionality is provided through the following packages.

.. rubric:: General Packages

*This section lists the core package that ties everything together as well as
other general packages and packages that don't fit it any of the other
categories.*

.. autosummary::
    :toctree: .
    :nosignatures:

    obspy.core
    obspy.geodetics
    obspy.imaging
    obspy.realtime
    obspy.signal
    obspy.taup


.. rubric:: Scripts

*All command-line scripts shipping with ObsPy.*

.. autosummary::
    :template: script.rst
    :toctree: autogen
    :nosignatures:

    obspy.scripts.flinnengdahl
    obspy.scripts.runtests
    obspy.scripts.reftekrescue
    obspy.scripts.print
    obspy.scripts.sds_html_report
    obspy.imaging.scripts.scan
    obspy.imaging.scripts.plot
    obspy.imaging.scripts.mopad
    obspy.io.mseed.scripts.recordanalyzer
    obspy.io.xseed.scripts.dataless2xseed
    obspy.io.xseed.scripts.xseed2dataless
    obspy.io.xseed.scripts.dataless2resp


.. rubric:: Database or Web Service Access Clients

*All ObsPy clients enabling remote and local access to data.*

.. autosummary::
    :toctree: .
    :nosignatures:

    obspy.clients.earthworm
    obspy.clients.fdsn
    obspy.clients.filesystem
    obspy.clients.iris
    obspy.clients.neic
    obspy.clients.nrl
    obspy.clients.seedlink
    obspy.clients.syngine


.. rubric:: Waveform Import/Export Plug-ins

.. warning::
    In most cases these modules do not need to be called directly. They
    register via the central ObsPy
    :func:`~obspy.core.stream.read` function - call this instead.


**Usage Example:**

.. code-block:: python

    import obspy
    # Format will be detected automatically.
    st = obspy.read("/path/to/file")
    # Many formats can also be written out - just use the module name.
    st.write("/path/to/outfile", format="mseed")


.. autosummary::
    :toctree: .
    :nosignatures:

    obspy.io.ah
    obspy.io.alsep
    obspy.io.ascii
    obspy.io.css
    obspy.io.dmx
    obspy.io.gcf
    obspy.io.gse2
    obspy.io.kinemetrics
    obspy.io.mseed
    obspy.io.nied.knet
    obspy.io.pdas
    obspy.io.reftek
    obspy.io.rg16
    obspy.io.sac
    obspy.io.seisan
    obspy.io.seg2
    obspy.io.segy
    obspy.io.sh
    obspy.io.wav
    obspy.io.win
    obspy.io.y

.. rubric:: Event Data Import/Export Plug-ins

.. warning::
    In most cases these modules do not need to be called directly. They
    register via the central ObsPy
    :func:`~obspy.core.event.read_events` function - call this instead.


**Usage Example:**

.. code-block:: python

    import obspy
    # Format will be detected automatically.
    cat = obspy.read_events("/path/to/file")
    # Many formats can also be written out - just use the module name.
    cat.write("/path/to/outfile", format="quakeml")

.. autosummary::
    :toctree: .
    :nosignatures:

    obspy.io.cmtsolution
    obspy.io.cnv
    obspy.io.focmec
    obspy.io.gse2
    obspy.io.hypodd
    obspy.io.iaspei
    obspy.io.json
    obspy.io.kml
    obspy.io.ndk
    obspy.io.nied.fnetmt
    obspy.io.nied.knet
    obspy.io.nlloc
    obspy.io.nordic
    obspy.io.pde
    obspy.io.quakeml
    obspy.io.scardec
    obspy.io.seiscomp
    obspy.io.shapefile
    obspy.io.zmap

.. rubric:: Inventory Data Import/Export Plug-ins


.. warning::
    In most cases these modules do not need to be called directly. They
    register via the central ObsPy
    :func:`~obspy.core.inventory.inventory.read_inventory` function -
    call this instead.


**Usage Example:**

.. code-block:: python

    import obspy
    # Format will be detected automatically.
    inv = obspy.read_inventory("/path/to/file")
    # Many formats can also be written out - just use the module name.
    inv.write("/path/to/outfile", format="stationxml")

.. autosummary::
    :toctree: .
    :nosignatures:

    obspy.io.css
    obspy.io.kml
    obspy.io.sac.sacpz
    obspy.io.seiscomp
    obspy.io.shapefile
    obspy.io.stationtxt
    obspy.io.stationxml
    obspy.io.xseed
