=================================
Seismometer Correction/Simulation
=================================

----------------------
Using a PAZ dictionary
----------------------

The following script shows how to simulate a 1Hz seismometer from a STS-2
seismometer with the given poles and zeros. Poles, zeros, gain
(*A0 normalization factor*) and sensitivity (*overall sensitivity*) are
specified as keys of a dictionary. 

.. include:: seismometer_correction_simulation_1.py
   :literal:

.. plot:: tutorial/code_snippets/seismometer_correction_simulation_1.py

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

-----------------
Using a RESP file
-----------------

It is further possible to use evalresp_ to evaluate the instrument
response information from a RESP file.

.. include:: seismometer_correction_simulation_3.py
   :literal:

.. plot:: tutorial/code_snippets/seismometer_correction_simulation_3.py

A :class:`~obspy.xseed.parser.Parser` object created using a Dataless SEED file
can also be used. For each trace the respective RESP response data is extracted
internally then. When using
:class:`~obspy.core.stream.Stream`/:class:`~obspy.core.trace.Trace`'s
:meth:`~obspy.core.trace.Trace.simulate` convenience methods the "date"
parameter can be omitted (each trace's start time is used internally).

.. include:: seismometer_correction_simulation_4.py
   :literal:

.. _matplotlib: http://matplotlib.sourceforge.net/

.. _evalresp: http://www.iris.edu/software/downloads/seed_tools/
