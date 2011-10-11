=======================
Trigger/Picker Tutorial
=======================

This is a small tutorial for the UNESCO short course on triggering. Test data
used in this tutorial can be downloaded here:
`trigger_data.zip <http://examples.obspy.org/trigger_data.zip>`_.

The triggers are implemented as described in [Withers1998]_. There
are two scripts in the repository which we used for detection in the Bavarian
network. They might be useful as reference code examples or serve as cookbook
for similar applications.

* `branches/sandbox/stalta/stalta4baynet.py <http://svn.obspy.org/branches/sandbox/stalta/stalta4baynet.py>`_
* `branches/sandbox/stalta/coincidence4baynet.py <http://svn.obspy.org/branches/sandbox/stalta/coincidence4baynet.py>`_


.. seealso::
    Please note the convenience method of ObsPy's
    :meth:`Stream.trigger <obspy.core.stream.Stream.trigger>` and
    :meth:`Trace.trigger <obspy.core.trace.Trace.trigger>`
    objects for triggering.

---------------------
Reading Waveform Data
---------------------

The data files are read into an ObsPy :class:`~obspy.core.trace.Trace` object
using the :func:`~obspy.core.stream.read()` function.

    >>> from obspy.core import read
    >>> st = read("http://examples.obspy.org/ev0_6.a01.gse2")
    >>> st = st.select(component="Z")
    >>> tr = st[0]

The data format is automatically detected. Important in this tutorial are the
:class:`~obspy.core.trace.Trace` attributes:
    
    ``tr.data``
        contains the data as :class:`numpy.ndarray`
        
    ``tr.stats``
        contains a dict-like class of header entries
    
    ``tr.stats.sampling_rate``
        the sampling rate
    
    ``tr.stats.npts``
        sample count of data

As an example, the header of the data file is printed and the data are plotted
like this:

    >>> print tr.stats
             network: 
             station: EV0_6
            location: 
             channel: EHZ
           starttime: 1970-01-01T01:00:00.000000Z
             endtime: 1970-01-01T01:00:59.995000Z
       sampling_rate: 200.0
               delta: 0.005
                npts: 12000
               calib: 1.0
             _format: GSE2
                gse2: AttribDict({'instype': '      ', 'datatype': 'CM6', 'hang': 0.0, 'auxid': '    ', 'vang': -1.0, 'calper': 1.0})

Using the :meth:`~obspy.core.trace.Trace.plot` method of the
:class:`~obspy.core.trace.Trace` objects will show the plot.

    >>> tr.plot(type="relative")

.. plot:: source/tutorial/trigger_tutorial.py

-----------------
Available Methods
-----------------

After loading the data, we are able to pass the waveform data to the following
trigger routines defined in :mod:`obspy.signal.trigger`:

    .. autosummary::
       :toctree: ../packages/autogen

       ~obspy.signal.trigger.recStalta
       ~obspy.signal.trigger.recStaltaPy
       ~obspy.signal.trigger.carlStaTrig
       ~obspy.signal.trigger.classicStaLta
       ~obspy.signal.trigger.delayedStaLta
       ~obspy.signal.trigger.zdetect
       ~obspy.signal.trigger.pkBaer
       ~obspy.signal.trigger.arPick

Help for each function is available  HTML formatted or in the usual Python manner:

    >>> from obspy.signal.trigger import classicStaLta
    >>> help(classicStaLta)  # doctest: +ELLIPSIS
    Help on function classicStaLta in module obspy.signal.trigger ...

The triggering itself mainly consists of the following two steps:

* Calculating the characteristic function
* Setting picks based on values of the characteristic function 

----------------
Trigger Examples
----------------

For all the examples, the commands to read in the data and to load the modules
are the following:

    >>> from obspy.core import read
    >>> from obspy.signal.trigger import *
    >>> from obspy.imaging.waveform import plot_trigger
    >>> trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
    >>> df = trace.stats.sampling_rate

Classic Sta Lta
===============

    >>> cft = classicStaLta(trace.data, int(5 * df), int(10 * df))
    >>> plot_trigger(trace, cft, 1.5, 0.5)

.. plot:: source/tutorial/trigger_tutorial_classic_sta_lta.py

Z-Detect
========

    >>> cft = zdetect(trace.data, int(10 * df))
    >>> plot_trigger(trace, cft, -0.4, -0.3)

.. plot:: source/tutorial/trigger_tutorial_z_detect.py

Recursive Sta Lta
=================

    >>> cft = recStaltaPy(trace.data, int(5 * df), int(10 * df))
    >>> plot_trigger(trace, cft, 1.2, 0.5)

.. plot:: source/tutorial/trigger_tutorial_recursive_sta_lta.py

Carl-Sta-Trig
=============

    >>> cft = carlStaTrig(trace.data, int(5 * df), int(10 * df), 0.8, 0.8)
    >>> plot_trigger(trace, cft, 20.0, -20.0)

.. plot:: source/tutorial/trigger_tutorial_carl_sta_trig.py

Delayed Sta Lta
===============

    >>> cft = delayedStaLta(trace.data, int(5 * df), int(10 * df))
    >>> plot_trigger(trace, cft, 5, 10)

.. plot:: source/tutorial/trigger_tutorial_delayed_sta_lta.py

----------------
Picker Examples
----------------

Baer Picker
===========

For :func:`~obspy.signal.trigger.pkBaer`, input is in seconds, output is in
samples.

    >>> from obspy.core import read
    >>> from obspy.signal.trigger import pkBaer
    >>> trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
    >>> df = trace.stats.sampling_rate
    >>> p_pick, phase_info = pkBaer(trace.data, df
    ...                             20, 60, 7.0, 12.0, 100, 100)
    >>> print p_pick, phase_info
    (6894, u'EPU3')
    >>> print(p_pick / df)
    34.47

This yields the output 34.47 EPU3, which means that a P pick was
set at 34.47s with Phase information EPU3.

AR Picker
=========

For :func:`~obspy.signal.trigger.arPick`, input and output are in seconds.

    >>> from obspy.core import read
    >>> from obspy.signal.trigger import arPick
    >>> tr1 = read('http://examples.obspy.org/loc_RJOB20050801145719850.z.gse2')[0]
    >>> tr2 = read('http://examples.obspy.org/loc_RJOB20050801145719850.n.gse2')[0]
    >>> tr3 = read('http://examples.obspy.org/loc_RJOB20050801145719850.e.gse2')[0]
    >>> df = tr1.stats.sampling_rate
    >>> p_pick, s_pick = arPick(tr1.data, tr2.data, tr3.data, df,
    ...                         1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
    >>> print p_pick, s_pick
    (30.6350002289 31.1499996185)

This gives the output 30.6350002289 31.1499996185, meaning that a P pick at
30.64s and an S pick at 31.15s were identified.

----------------
Advanced Example
----------------

A more complicated example, where the data are retrieved via ArcLink and
results are plotted step by step, is shown here:

.. include:: trigger_tutorial_advanced.py
   :literal:

.. plot:: source/tutorial/trigger_tutorial_advanced.py
