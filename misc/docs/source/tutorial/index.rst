.. _tutorial:

Tutorial
========

.. note::
    A one-hour introduction to ObsPy is
    `available at YouTube <https://www.youtube.com/watch?v=kFwdjfiK4gk>`__.

This tutorial does not attempt to be comprehensive and cover every single
feature. Instead, it introduces many of ObsPy's most noteworthy features, and
will give you a good idea of the library’s flavor and style.

A Chinese translation of the tutorial (as of 2020-04-12) is available `here <http://docs.obspy.org/archive/ObsPy_Tutorial_2020-04_chinese.pdf>`_.

There are also IPython notebooks available online with an
`introduction to Python <https://nbviewer.jupyter.org/github/obspy/docs/blob/master/workshops/2017-10-25_iris_stcu/Python%20Introduction/Python_Crash_Course.ipynb>`__
(`with solutions/output <https://nbviewer.jupyter.org/github/obspy/docs/blob/master/workshops/2017-10-25_iris_stcu/Python%20Introduction/Python_Crash_Course-with_solutions.ipynb>`__),
an
`introduction to ObsPy split up in multiple chapters <http://nbviewer.jupyter.org/github/obspy/docs/blob/master/workshops/2017-10-25_iris_stcu/ObsPy%20Tutorial/00_Introduction.ipynb>`__ (again, versions with/without solutions available)
and a
`brief primer on data center access and visualization with ObsPy <https://nbviewer.jupyter.org/github/obspy/docs/blob/master/notebooks/Direct_Access_to_Seismological_Data_using_Python_and_ObsPy.ipynb>`__.
There are also nice `Jupyter notebooks with an introduction to matplotlib <http://nbviewer.jupyter.org/github/matplotlib/AnatomyOfMatplotlib/tree/master/>`__.

Introduction to ObsPy
---------------------

.. toctree::
   :maxdepth: 2

   code_snippets/python_introduction
   code_snippets/utc_date_time
   code_snippets/reading_seismograms
   code_snippets/waveform_plotting_tutorial
   code_snippets/retrieving_data_from_datacenters
   code_snippets/filtering_seismograms
   code_snippets/downsampling_seismograms
   code_snippets/merging_seismograms
   code_snippets/beamforming_fk_analysis
   code_snippets/seismogram_envelopes
   code_snippets/plotting_spectrograms
   code_snippets/trigger_tutorial
   code_snippets/frequency_response
   code_snippets/seismometer_correction_simulation
   code_snippets/clone_dataless_seed
   code_snippets/export_seismograms_to_matlab
   code_snippets/export_seismograms_to_ascii
   code_snippets/anything_to_miniseed
   code_snippets/beachball_plot
   code_snippets/cartopy_plot_with_beachballs
   code_snippets/interfacing_r_from_python
   code_snippets/coordinate_conversions
   code_snippets/hierarchical_clustering.rst
   code_snippets/probabilistic_power_spectral_density
   code_snippets/array_response_function
   code_snippets/continuous_wavelet_transform
   code_snippets/time_frequency_misfit
   code_snippets/visualize_data_availability_of_local_waveform_archive
   code_snippets/travel_time
   code_snippets/xcorr_pick_correction
   code_snippets/xcorr_detector
   code_snippets/quakeml_custom_tags
   code_snippets/stationxml_custom_tags
   code_snippets/stationxml_file_from_scratch
   code_snippets/easyseedlink


Advanced Exercise
-----------------

In the advanced exercise we show how ObsPy can be used to develop an automated
processing workflow. We start out with very simple tasks and then automate the
routine step by step.
For all exercises solutions are provided.

.. toctree::
   :maxdepth: 1

   advanced_exercise/advanced_exercise

.. _code-snippets:
