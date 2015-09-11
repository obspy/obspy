=================================
Seismometer Correction/Simulation
=================================

--------------------------------------------------------
Calculating response from filter stages using evalresp..
--------------------------------------------------------

..using a StationXML file or in general an :class:`~obspy.station.inventory.Inventory` object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using the :class:`FDSN client <obspy.fdsn.client.Client>` the response can
directly be attached to the waveforms and then subsequently removed using
:meth:`Stream.remove_response() <obspy.core.stream.Stream.remove_response>`:

.. code-block:: python

    from obspy import UTCDateTime
    from obspy.fdsn import Client

    t1 = UTCDateTime("2010-09-3T16:30:00.000")
    t2 = UTCDateTime("2010-09-3T17:00:00.000")
    fdsn_client = Client('IRIS')
    # Fetch waveform from IRIS FDSN web service into a ObsPy stream object
    # and automatically attach correct response
    st = fdsn_client.get_waveforms(network='NZ', station='BFZ', location='10',
                                   channel='HHZ', starttime=t1, endtime=t2,
                                   attach_response=True)
    # define a filter band to prevent amplifying noise during the deconvolution
    pre_filt = (0.005, 0.006, 30.0, 35.0)
    st.remove_response(output='DISP', pre_filt=pre_filt)

Alternatively an :class:`~obspy.station.inventory.Inventory` object can be used
to attach response information:


.. code-block:: python

    from obspy import read, read_inventory

    # simply use the included example waveform
    st = read()
    # the corresponding response is included in ObsPy as a StationXML file
    inv = read_inventory("/path/to/BW_RJOB.xml")
    # the helper routine automatically picks the correct response for each trace
    # the routine returns a list of traces for which no matching response could
    # be found in the inventory
    st.attach_response(inv)
    # define a filter band to prevent amplifying noise during the deconvolution
    pre_filt = (0.005, 0.006, 30.0, 35.0)
    st.remove_response(output='DISP', pre_filt=pre_filt)

Using the `plot` option it is possible to visualize the individual steps during
response removal in the frequency domain to check the chosen `pre_filt` and
`water_level` options to stabilize the deconvolution of the inverted instrument
response spectrum:

.. plot:: tutorial/code_snippets/seismometer_correction_simulation_5.py
   :include-source:

..using a RESP file
^^^^^^^^^^^^^^^^^^^

It is further possible to use evalresp_ to evaluate the instrument
response information from a RESP file.

.. plot:: tutorial/code_snippets/seismometer_correction_simulation_3.py
   :include-source:

..using a Dataless/Full SEED file (or XMLSEED file)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~obspy.xseed.parser.Parser` object created using a Dataless SEED file
can also be used. For each trace the respective RESP response data is extracted
internally then. When using
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`'s
:meth:`~obspy.core.trace.Trace.simulate` convenience methods the "date"
parameter can be omitted (each trace's start time is used internally).

.. include:: seismometer_correction_simulation_4.py
   :literal:

----------------------
Using a PAZ dictionary
----------------------

The following script shows how to simulate a 1Hz seismometer from a STS-2
seismometer with the given poles and zeros. Poles, zeros, gain
(*A0 normalization factor*) and sensitivity (*overall sensitivity*) are
specified as keys of a dictionary.

.. plot:: tutorial/code_snippets/seismometer_correction_simulation_1.py
   :include-source:

For more customized plotting we could also work with matplotlib_ manually from
here:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   tr = st[0]
   tr_orig = st_orig[0]

   t = np.arange(tr.stats.npts) / tr.stats.sampling_rate

   plt.subplot(211)
   plt.plot(t, tr_orig.data, 'k')
   plt.ylabel('STS-2 [counts]')
   plt.subplot(212)
   plt.plot(t, tr.data, 'k')
   plt.ylabel('1Hz Instrument [m/s]')
   plt.xlabel('Time [s]')
   plt.show()

.. plot:: tutorial/code_snippets/seismometer_correction_simulation_2.py

.. _matplotlib: http://matplotlib.org/

.. _evalresp: http://www.iris.edu/ds/nodes/dmc/software/downloads/evalresp/
