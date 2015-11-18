=======================
ObsPy Library Reference
=======================

The functionality is provided through the following packages:

.. rubric:: General Packages

.. autosummary::
   :toctree: .
   :nosignatures:

   obspy.core
   obspy.geodetics
   obspy.imaging
   obspy.realtime
   obspy.signal
   obspy.taup
   obspy.io.xseed

.. rubric:: Scripts

.. autosummary::
   :template: script.rst
   :toctree: autogen
   :nosignatures:

   obspy.scripts.runtests
   obspy.scripts.reftekrescue

.. rubric:: Waveform Import/Export Plug-ins

.. autosummary::
   :toctree: .
   :nosignatures:

   obspy.io.ah
   obspy.io.ascii
   obspy.io.css
   obspy.io.datamark
   obspy.io.gse2
   obspy.io.kinemetrics
   obspy.io.mseed
   obspy.io.nied.knet
   obspy.io.pdas
   obspy.io.sac
   obspy.io.seisan
   obspy.io.seg2
   obspy.io.segy
   obspy.io.sh
   obspy.io.wav
   obspy.io.y

.. rubric:: Event Data Import/Export Plug-ins

.. autosummary::
   :toctree: .
   :nosignatures:

   obspy.io.cmtsolution
   obspy.io.cnv
   obspy.io.json
   obspy.io.kml
   obspy.io.ndk
   obspy.io.nied.fnetmt
   obspy.io.nlloc
   obspy.io.pde
   obspy.io.shapefile
   obspy.io.quakeml
   obspy.io.zmap

.. rubric:: Inventory Data Import/Export Plug-ins

.. autosummary::
   :toctree: .
   :nosignatures:

   obspy.io.css
   obspy.io.kml
   obspy.io.sac.sacpz
   obspy.io.shapefile
   obspy.io.stationtxt
   obspy.io.stationxml

.. rubric:: Database or Web Service Access Clients

.. autosummary::
   :toctree: .
   :nosignatures:

   obspy.clients.arclink
   obspy.clients.earthworm
   obspy.clients.fdsn
   obspy.clients.filesystem
   obspy.clients.iris
   obspy.clients.neic
   obspy.clients.seedlink
   obspy.clients.seishub
   obspy.db
