=========================
Beamforming - FK Analysis
=========================

The following code shows how to do an FK Analysis with ObsPy. The data are from
the blasting of the AGFA skyscraper in Munich (:meth:`pickle.pickle.dump` stream
object with paz attached to each :class:`~obspy.core.trace.Trace`, can also be
found `here <http://examples.obspy.org/agfa.dump>`_). We execute
:func:`~obspy.signal.array_analysis.sonic` using the following settings:

* The slowness grid is set to corner values of -3.0 to 3.0 s/km with a step
  fraction of ``sl_s = 0.03``.
* The window length is 1.0 s, using a step fraction of 0.05 s.
* The data is bandpass filtered, using corners at 1.0 and 8.0 Hz,
  prewhitening is disabled.
* ``semb_thres`` and ``vel_thres`` are set to infinitesimally small numbers
  and must not be changed.
* The ``timestamp`` will be written in ``'mlabhours'``, which can be read
  directly by our plotting routine.
* ``stime`` and ``etime`` have to be given in the UTCDateTime format.

The output will be stored in ``out``.

.. code-block:: python

    from obspy.core import UTCDateTime
    from obspy.signal import cornFreq2Paz
    from obspy.signal.array_analysis import sonic
    import pickle
    import urllib
    
    # Load data
    st = pickle.load(urllib.urlopen("http://examples.obspy.org/agfa.dump"))
    
    # Instrument correction to 1Hz corner frequency
    paz1hz = cornFreq2Paz(1.0, damp=0.707)
    st.simulate(paz_remove='self', paz_simulate=paz1hz)
    
    # Execute sonic
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
        # sliding window propertieds
        win_len=1.0, win_frac=0.05,
        # frequency properties
        frqlow=1.0, frqhigh=8.0, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, verbose=True, timestamp='mlabhour',
        stime=UTCDateTime("20080217110515"), etime=UTCDateTime("20080217110545")
    )
    out = sonic(st, **kwargs)

The output can be plotted by the following code lines. We read the output
``out`` produced by :func:`~obspy.signal.array_analysis.sonic`, which are
:class:`numpy.ndarrays <numpy.ndarray>` containing timestamp, relative power,
absolute power, backazimuth, slowness. The colorbar corresponds to relative
power.

.. code-block:: python

    import matplotlib.pyplot as plt
    labels = 'rel.power abs.power baz slow'.split()
    
    fig = plt.figure()
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(4, 1, i + 1)
        ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                   edgecolors='none')
        ax.set_ylabel(lab)
        ax.xaxis_date()
    
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0)

.. plot:: source/tutorial/beamforming_fk_analysis_1.py

Another representation would be a polar plot, which sums the relative power in
gridded bins, each defined by backazimuth and slowness of the analyzed signal
part. The backazimuth is counted clockwise from north, the slowness limits can
be set by hand.

.. code-block:: python

    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np
    
    cmap = cm.hot_r
    pi = np.pi
    
    #
    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    
    # choose number of fractions in plot (desirably 360 degree/N is an integer!)
    N = 30
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 3, N + 1)
    
    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = np.histogram2d(baz, slow,
            bins=[abins, sbins], weights=rel_power)
    
    # transform to gradient
    baz_edges = baz_edges / 180 * np.pi
    
    # add polar and colorbar axes
    fig = plt.figure(figsize=(8, 8))
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
    
    dh = abs(sl_edges[1] - sl_edges[0])
    dw = abs(baz_edges[1] - baz_edges[0])
    
    # circle through backazimuth
    for i, row in enumerate(hist):
        bars = ax.bar(left=(pi / 2 - (i + 1) * dw) * np.ones(N),
                      height=dh * np.ones(N),
                      width=dw, bottom=dh * np.arange(N),
                      color=cmap(row / hist.max()))
    
    ax.set_xticks([pi / 2, 0, 3. / 2 * pi, pi])
    ax.set_xticklabels(['N', 'E', 'S', 'W'])
    
    # set slowness limits
    ax.set_ylim(0, 3)
    ColorbarBase(cax, cmap=cmap,
                 norm=Normalize(vmin=hist.min(), vmax=hist.max()))
    
    plt.show()

.. plot:: source/tutorial/beamforming_fk_analysis_2.py
