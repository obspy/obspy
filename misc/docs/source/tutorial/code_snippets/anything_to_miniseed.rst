====================
Anything to MiniSEED
====================

The following lines show how you can convert anything to MiniSEED_ format. In
the example, a few lines of a weather station output are written to a MiniSEED
file. The correct meta information ``starttime``, the ``sampling_rate``, 
``station`` name and so forth are also encoded (Note: Only the ones given are
allowed by the MiniSEED standard). Converting arbitrary ASCII to MiniSEED is
extremely helpful if you want to send log messages, output of meteorologic
stations or anything else via the SeedLink_ protocol.

.. include:: anything_to_miniseed.py
   :literal:

.. _MiniSEED: https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf
.. _SeedLink: https://ds.iris.edu/ds/nodes/dmc/services/seedlink/
