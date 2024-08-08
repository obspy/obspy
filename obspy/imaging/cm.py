# -*- coding: utf-8 -*-
"""
Module for ObsPy's default colormaps.

Overview of provided colormaps:
===============================

The following colormaps can be imported like..

    >>> from obspy.imaging.cm import viridis_r

List of all colormaps:

    * `viridis`_
    * `viridis_r`_
    * `viridis_white`_
    * `viridis_white_r`_
    * obspy_sequential (alias for `viridis`_)
    * obspy_sequential_r (alias for `viridis_r`_)
    * obspy_divergent (alias for matplotlib's RdBu_r)
    * obspy_divergent_r (alias for matplotlib's RdBu)
    * `pqlx`_

.. plot::

    from obspy.imaging.cm import _colormap_plot_overview
    _colormap_plot_overview()

viridis
-------

"viridis" is matplotlib's new default colormap from version 2.0 onwards and is
based on a design by Eric Firing (@efiring, see
http://thread.gmane.org/gmane.comp.python.matplotlib.devel/13522/focus=13542).

    >>> from obspy.imaging.cm import viridis

.. plot::

    from obspy.imaging.cm import viridis as cmap
    from obspy.imaging.cm import _colormap_plot_cwt as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis as cmap
    from obspy.imaging.cm import _colormap_plot_array_response as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis as cmap
    from obspy.imaging.cm import _colormap_plot_ppsd as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis as cmap
    from obspy.imaging.cm import _colormap_plot_similarity as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis as cmap
    from obspy.imaging.cm import _colormap_plot_beamforming_time as plot
    plot([cmap])

viridis_r
---------

Reversed version of viridis.

    >>> from obspy.imaging.cm import viridis_r

.. plot::

    from obspy.imaging.cm import viridis_r as cmap
    from obspy.imaging.cm import _colormap_plot_cwt as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_r as cmap
    from obspy.imaging.cm import _colormap_plot_array_response as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_r as cmap
    from obspy.imaging.cm import _colormap_plot_ppsd as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_r as cmap
    from obspy.imaging.cm import _colormap_plot_similarity as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_r as cmap
    from obspy.imaging.cm import _colormap_plot_beamforming_time as plot
    plot([cmap])

viridis_white
-------------

"viridis_white" is a modified version of "viridis" that goes to white instead
of yellow in the end. Although it remains perceptually uniform, the light
colors are a bit more difficult to distinguish than yellow in the original
viridis. It is useful for printing because one end of the colorbar can merge
with a white background (by M Meschede).

    >>> from obspy.imaging.cm import viridis_white

.. plot::

    from obspy.imaging.cm import viridis_white as cmap
    from obspy.imaging.cm import _colormap_plot_cwt as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white as cmap
    from obspy.imaging.cm import _colormap_plot_array_response as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white as cmap
    from obspy.imaging.cm import _colormap_plot_ppsd as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white as cmap
    from obspy.imaging.cm import _colormap_plot_similarity as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white as cmap
    from obspy.imaging.cm import _colormap_plot_beamforming_time as plot
    plot([cmap])

viridis_white_r
---------------

Reversed version of viridis_white.

    >>> from obspy.imaging.cm import viridis_white_r

.. plot::

    from obspy.imaging.cm import viridis_white_r as cmap
    from obspy.imaging.cm import _colormap_plot_cwt as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white_r as cmap
    from obspy.imaging.cm import _colormap_plot_array_response as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white_r as cmap
    from obspy.imaging.cm import _colormap_plot_ppsd as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white_r as cmap
    from obspy.imaging.cm import _colormap_plot_similarity as plot
    plot([cmap])

.. plot::

    from obspy.imaging.cm import viridis_white_r as cmap
    from obspy.imaging.cm import _colormap_plot_beamforming_time as plot
    plot([cmap])

pqlx
----

Colormap defined and used in PQLX (see [McNamara2004]_).

    >>> from obspy.imaging.cm import pqlx

.. plot::

    from obspy.imaging.cm import pqlx as cmap
    from obspy.imaging.cm import _colormap_plot_ppsd as plot
    plot([cmap])

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import glob
import inspect
import io
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import get_cmap


def _get_cmap(file_name, lut=None, reverse=False):
    """
    Load a :class:`~matplotlib.colors.LinearSegmentedColormap` from
    `segmentdata` dictionary saved as numpy compressed binary data.

    :type file_name: str
    :param file_name: Name of colormap to load, same as filename in
        `obspy/imaging/data`. The type of colormap data is determined from the
        extension: .npz assumes the file contains colorbar segments (segmented
        colormap). '*.npy' assumes the file contains a simple array of RGB
        values with size [ncolors, 3].
    :type lut: int
    :param lut: Specifies the number of discrete color values in the segmented
        colormap. Only used for segmented colormap `None` to use matplotlib
        default value (continuous colormap).
    :type reverse: bool
    :param reverse: Whether to return the specified colormap reverted.
    :rtype: :class:`~matplotlib.colors.LinearSegmentedColormap`
    """
    file_name = file_name.strip()
    file_path = Path(file_name)
    name = str(file_path.parent / file_path.stem)
    suffix = file_path.suffix
    directory = Path(inspect.getfile(
                                    inspect.currentframe()))
    directory = directory.resolve().parent / "data"
    full_path = directory / file_name
    # check if it is npz -> segmented colormap or npy -> listed colormap
    # do it like matplotlib, append "_r" to reverted versions
    if reverse:
        name += "_r"

    if suffix == '.npz':
        # segmented colormap
        data = dict(np.load(full_path))
        if reverse:
            data_r = {}
            for key, val in data.items():
                # copied from matplotlib source,
                # cm.py@f7a578656abc2b2c13 line 47
                data_r[key] = [(1.0 - x, y1, y0) for x, y0, y1 in
                               reversed(val)]
            data = data_r
        kwargs = lut and {"N": lut} or {}
        cmap = LinearSegmentedColormap(name=name, segmentdata=data, **kwargs)
    elif suffix == '.npy':
        # listed colormap
        data = np.load(full_path)
        if reverse:
            data = data[::-1]
        cmap = ListedColormap(data, name=name)
    else:
        raise ValueError('file suffix {} not recognized.'.format(suffix))

    return cmap


def _get_all_cmaps():
    """
    Return all colormaps in "obspy/imaging/data" directory, including reversed
    versions.

    :rtype: dict
    """
    cmaps = {}
    cm_file_pattern = Path(inspect.getfile(
                                            inspect.currentframe()))
    cm_file_pattern = str(cm_file_pattern.parent.resolve()/"data" / "*.np[yz]")
    for filename in glob.glob(cm_file_pattern):
        filename = Path(filename).name
        for reverse in (True, False):
            # don't add a reversed version for PQLX colormap
            if filename == "pqlx.npz" and reverse:
                continue
            cmap = _get_cmap(filename, reverse=reverse)
            cmaps[cmap.name] = cmap
    return cmaps


# inject all colormaps into namespace
_globals = globals()
_globals.update(_get_all_cmaps())

obspy_sequential = _globals["viridis"]
obspy_sequential_r = _globals["viridis_r"]
obspy_divergent = get_cmap("RdBu_r")
obspy_divergent_r = get_cmap("RdBu")
#: PQLX colormap
pqlx = _get_cmap("pqlx.npz")


def _colormap_plot_overview(colormap_names=(
        "viridis", "obspy_sequential", "viridis_white", "viridis_r",
        "obspy_sequential_r", "viridis_white_r", "obspy_divergent",
        "obspy_divergent_r", "pqlx")):
    """
    Overview bar plot, adapted after
    http://scipy-cookbook.readthedocs.org/items/Matplotlib_Show_colormaps.html.
    """
    import matplotlib.pyplot as plt
    import importlib
    cm = importlib.import_module("obspy.imaging.cm")
    plt.rc('text', usetex=False)
    a = np.outer(np.ones(1000), np.linspace(0, 1, 1000))
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.11, right=0.89)
    extent = (0, 1, 0, 1)
    for i, name in enumerate(colormap_names):
        cmap = getattr(cm, name)
        ax = fig.add_subplot(len(colormap_names), 1, i + 1)
        ax.imshow(a, aspect='auto', cmap=cmap, origin="lower", extent=extent,
                  interpolation="nearest")
        ax.set_ylabel(name, family="monospace", fontsize="large", ha="right",
                      rotation="horizontal")
    for ax in fig.axes:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.yaxis.set_ticks_position("none")
    for ax in fig.axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    plt.show()


def _colormap_plot_ppsd(cmaps):
    """
    Plot for illustrating colormaps: PPSD.

    :param cmaps: list of :class:`~matplotlib.colors.Colormap`
    :rtype: None
    """
    import matplotlib.pyplot as plt
    from obspy import read
    from obspy.signal import PPSD
    from obspy.io.xseed import Parser
    st = read("https://examples.obspy.org/BW.KW1..EHZ.D.2011.037")
    st += read("https://examples.obspy.org/BW.KW1..EHZ.D.2011.038")
    parser = Parser("https://examples.obspy.org/dataless.seed.BW_KW1")
    ppsd = PPSD(st[0].stats, metadata=parser)
    ppsd.add(st)

    for cmap in cmaps:
        ppsd.plot(cmap=cmap, show=False)
    plt.show()


def _colormap_plot_array_response(cmaps):
    """
    Plot for illustrating colormaps: array response.

    :param cmaps: list of :class:`~matplotlib.colors.Colormap`
    :rtype: None
    """
    import matplotlib.pyplot as plt
    from obspy.signal.array_analysis import array_transff_wavenumber
    # generate array coordinates
    coords = np.array([[10., 60., 0.], [200., 50., 0.], [-120., 170., 0.],
                       [-100., -150., 0.], [30., -220., 0.]])
    # coordinates in km
    coords /= 1000.
    # set limits for wavenumber differences to analyze
    klim = 40.
    kxmin = -klim
    kxmax = klim
    kymin = -klim
    kymax = klim
    kstep = klim / 100.
    # compute transfer function as a function of wavenumber difference
    transff = array_transff_wavenumber(coords, klim, kstep, coordsys='xy')
    # plot
    for cmap in cmaps:
        plt.figure()
        plt.pcolor(np.arange(kxmin, kxmax + kstep * 1.1, kstep) - kstep / 2.,
                   np.arange(kymin, kymax + kstep * 1.1, kstep) - kstep / 2.,
                   transff.T, cmap=cmap)
        plt.colorbar()
        plt.clim(vmin=0., vmax=1.)
        plt.xlim(kxmin, kxmax)
        plt.ylim(kymin, kymax)
    plt.show()


def _colormap_plot_cwt(cmaps):
    """
    Plot for illustrating colormaps: cwt.

    :param cmaps: list of :class:`~matplotlib.colors.Colormap`
    :rtype: None
    """
    import matplotlib.pyplot as plt
    from obspy import read
    from obspy.signal.tf_misfit import cwt
    tr = read()[0]
    npts = tr.stats.npts
    dt = tr.stats.delta
    t = np.linspace(0, dt * npts, npts)
    f_min = 1
    f_max = 50
    scalogram = cwt(tr.data, dt, 8, f_min, f_max)
    x, y = np.meshgrid(
        t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    for cmap in cmaps:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolormesh(x, y, np.abs(scalogram), cmap=cmap)
        ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_yscale('log')
        ax.set_ylim(f_min, f_max)
    plt.show()


def _colormap_plot_similarity(cmaps):
    """
    Plot for illustrating colormaps: similarity matrix.

    :param cmaps: list of :class:`~matplotlib.colors.Colormap`
    :rtype: None
    """
    import matplotlib.pyplot as plt

    url = "https://examples.obspy.org/dissimilarities.npz"
    with io.BytesIO(urlopen(url).read()) as fh, np.load(fh) as data:
        dissimilarity = data['dissimilarity']

    for cmap in cmaps:
        plt.figure(figsize=(6, 5))
        plt.subplot(1, 1, 1)
        plt.imshow(1 - dissimilarity, interpolation='nearest', cmap=cmap)
        plt.xlabel("Event number")
        plt.ylabel("Event number")
        cb = plt.colorbar()
        cb.set_label("Similarity")
    plt.show()


def _get_beamforming_example_stream():
    # Load data
    from obspy import read
    from obspy.core.util import AttribDict
    from obspy.signal.invsim import corn_freq_2_paz
    st = read("https://examples.obspy.org/agfa.mseed")
    # Set PAZ and coordinates for all 5 channels
    st[0].stats.paz = AttribDict({
        'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
        'zeros': [0j, 0j],
        'sensitivity': 205479446.68601453,
        'gain': 1.0})
    st[0].stats.coordinates = AttribDict({
        'latitude': 48.108589,
        'elevation': 0.450000,
        'longitude': 11.582967})
    st[1].stats.paz = AttribDict({
        'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
        'zeros': [0j, 0j],
        'sensitivity': 205479446.68601453,
        'gain': 1.0})
    st[1].stats.coordinates = AttribDict({
        'latitude': 48.108192,
        'elevation': 0.450000,
        'longitude': 11.583120})
    st[2].stats.paz = AttribDict({
        'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
        'zeros': [0j, 0j],
        'sensitivity': 250000000.0,
        'gain': 1.0})
    st[2].stats.coordinates = AttribDict({
        'latitude': 48.108692,
        'elevation': 0.450000,
        'longitude': 11.583414})
    st[3].stats.paz = AttribDict({
        'poles': [(-4.39823 + 4.48709j), (-4.39823 - 4.48709j)],
        'zeros': [0j, 0j],
        'sensitivity': 222222228.10910088,
        'gain': 1.0})
    st[3].stats.coordinates = AttribDict({
        'latitude': 48.108456,
        'elevation': 0.450000,
        'longitude': 11.583049})
    st[4].stats.paz = AttribDict({
        'poles': [(-4.39823 + 4.48709j), (-4.39823 - 4.48709j), (-2.105 + 0j)],
        'zeros': [0j, 0j, 0j],
        'sensitivity': 222222228.10910088,
        'gain': 1.0})
    st[4].stats.coordinates = AttribDict({
        'latitude': 48.108730,
        'elevation': 0.450000,
        'longitude': 11.583157})
    # Instrument correction to 1Hz corner frequency
    paz1hz = corn_freq_2_paz(1.0, damp=0.707)
    st.simulate(paz_remove='self', paz_simulate=paz1hz)
    return st


def _colormap_plot_beamforming_time(cmaps):
    """
    Plot for illustrating colormaps: beamforming.

    :param cmaps: list of :class:`~matplotlib.colors.Colormap`
    :rtype: None
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    from obspy import UTCDateTime
    from obspy.signal.array_analysis import array_processing

    # Execute array_processing
    stime = UTCDateTime("20080217110515")
    etime = UTCDateTime("20080217110545")
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
        # sliding window properties
        win_len=1.0, win_frac=0.05,
        # frequency properties
        frqlow=1.0, frqhigh=8.0, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        stime=stime, etime=etime
    )
    st = _get_beamforming_example_stream()
    out = array_processing(st, **kwargs)
    # Plot
    labels = ['rel.power', 'abs.power', 'baz', 'slow']
    xlocator = mdates.AutoDateLocator()
    for cmap in cmaps:
        fig = plt.figure()
        for i, lab in enumerate(labels):
            ax = fig.add_subplot(4, 1, i + 1)
            ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                       edgecolors='none', cmap=cmap)
            ax.set_ylabel(lab)
            ax.set_xlim(out[0, 0], out[-1, 0])
            ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
        fig.suptitle('AGFA skyscraper blasting in Munich %s' % (
            stime.strftime('%Y-%m-%d'), ))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2,
                            hspace=0)
    plt.show()


def _colormap_plot_beamforming_polar(cmaps):
    """
    Plot for illustrating colormaps: beamforming.

    :param cmaps: list of :class:`~matplotlib.colors.Colormap`
    :rtype: None
    """
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    from obspy import UTCDateTime
    from obspy.signal.array_analysis import array_processing
    # Execute array_processing
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
        # sliding window properties
        win_len=1.0, win_frac=0.05,
        # frequency properties
        frqlow=1.0, frqhigh=8.0, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9,
        stime=UTCDateTime("20080217110515"),
        etime=UTCDateTime("20080217110545")
    )
    st = _get_beamforming_example_stream()
    out = array_processing(st, **kwargs)
    # make output human readable, adjust backazimuth to values between 0 and
    # 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    # choose number of fractions in plot (desirably 360 degree/N is an
    # integer!)
    num = 36
    num2 = 30
    abins = np.arange(num + 1) * 360. / num
    sbins = np.linspace(0, 3, num2 + 1)
    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = \
        np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)
    # transform to radian
    baz_edges = np.radians(baz_edges)
    dh = abs(sl_edges[1] - sl_edges[0])
    dw = abs(baz_edges[1] - baz_edges[0])
    for cmap in cmaps:
        # add polar and colorbar axes
        fig = plt.figure(figsize=(8, 8))
        cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
        ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        # circle through backazimuth
        for i, row in enumerate(hist):
            ax.bar((i * dw) * np.ones(num2),
                   height=dh * np.ones(num2),
                   width=dw, bottom=dh * np.arange(num2),
                   color=cmap(row / hist.max()))
        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])
        # set slowness limits
        ax.set_ylim(0, 3)
        [i.set_color('grey') for i in ax.get_yticklabels()]
        ColorbarBase(cax, cmap=cmap,
                     norm=Normalize(vmin=hist.min(), vmax=hist.max()))
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
