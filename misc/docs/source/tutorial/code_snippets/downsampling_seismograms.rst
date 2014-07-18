========================
Downsampling Seismograms
========================

The following script shows how to downsample a seismogram. Currently, a simple
integer decimation is supported. If not explicitly disabled, a low-pass filter
is applied prior to decimation in order to prevent aliasing. For comparison,
the non-decimated but filtered data is plotted as well. Applied processing
steps are documented in ``trace.stats.processing`` of every single Trace. Note
the shift that is introduced because by default the applied filters are not of
zero-phase type. This can be avoided by manually applying a zero-phase filter
and deactivating automatic filtering during downsampling (``no_filter=True``).

.. plot:: tutorial/code_snippets/downsampling_seismograms.py
   :include-source:
