## ObsPy: A Python Toolbox for seismology/seismological observatories.

ObsPy is an open-source project dedicated to provide a **Python framework for processing seismological** data. It provides parsers for common file formats and seismological signal processing routines which allow the manipulation of seismological time series (see [Beyreuther et al. 2010](http://www.seismosoc.org/publications/SRL/SRL_81/srl_81-3_es/), [Megies et al. 2011](http://www.annalsofgeophysics.eu/index.php/annals/article/view/4838)).

The goal of the ObsPy project is to facilitate **rapid application development for seismology**.

## Getting Started

The [ObsPy Gallery](http://gallery.obspy.org) and its related [ObsPy Tutorial](http://tutorial.obspy.org) are maybe the best point to get a first impression of what ObsPy is all about. The tutorial is a collection of short example programs with explanations and program output. 

## Installing ObsPy

ObsPy is currently [running and tested](http://tests.obspy.org) on Linux (32 and 64 bit), Windows XP/Vista/7 (32 bit and/or 64 bit) and Mac OS X (32 and 64 bit Intel and untested support for 32 and 64 bit PPC).

These notes describe installing ObsPy on the following platforms:

* [[Linux|InstallationLinux]]
* [[Mac|InstallationMac]]
* [[Windows|InstallationWindows]]
* [[FreeBSD|InstallationFreeBSD]]

## Mailing List

If you are using ObsPy we **strongly recommend** for you to join the [[obspy-users] mailing list](http://lists.obspy.org/listinfo). This list will be the place where new additions, important changes and bug fixes will be announced. The list can also be used to contact other ObsPy users for open discussions.

## [Documentation](http://docs.obspy.org)

The functionality is provided through the following packages:

### General packages:
* [obspy.core](http://docs.obspy.org/packages/obspy.core.html) - ObsPy core package, glues the single obspy packages together
* [obspy.imaging](http://docs.obspy.org/packages/obspy.imaging.html) - Imaging spectograms, beachballs and waveforms
* [obspy.realtime](http://docs.obspy.org/packages/obspy.realtime.html) - Extends the !ObsPy core classes with real time functionalities (''experimental'')
* [obspy.signal](http://docs.obspy.org/packages/obspy.signal.html) - Filters, triggers, instrument correction, rotation, array analysis, beamforming
* [obspy.taup](http://docs.obspy.org/packages/obspy.taup.html) - Calculates and visualizes travel times
* [obspy.xseed](http://docs.obspy.org/packages/obspy.xseed.html) - Converter for Dataless SEED, [XML-SEED](http://adsabs.harvard.edu/abs/2004AGUFMSF31B..03T) and SEED RESP files

### Waveform import/export plug-ins:
* [obspy.datamark](http://docs.obspy.org/packages/obspy.datamark.html) - DATAMARK read support ("experimental")
* [obspy.gse2](http://docs.obspy.org/packages/obspy.gse2.html) - GSE2 and GSE1 read and write support
* [obspy.mseed](http://docs.obspy.org/packages/obspy.mseed.html) - MiniSEED read and write support
* [obspy.sac](http://docs.obspy.org/packages/obspy.sac.html) - SAC read and write support
* [obspy.seisan](http://docs.obspy.org/packages/obspy.seisan.html) - SEISAN read support
* [obspy.seg2](http://docs.obspy.org/packages/obspy.seg2.html) - SEG2 read support ("experimental")
* [obspy.segy](http://docs.obspy.org/packages/obspy.segy.html) - SEGY read and write support
* [obspy.sh](http://docs.obspy.org/packages/obspy.sh.html) - Q and ASC read and write support (file formats of [SeismicHandler](http://www.seismic-handler.org))
* [obspy.wav](http://docs.obspy.org/packages/obspy.wav.html) - WAV (audio) read and write support

### Database or Web service access clients:
* [obspy.arclink](http://docs.obspy.org/packages/obspy.arclink.html) - [ArcLink/WebDC](http://www.webdc.eu) request client
* [obspy.earthworm](http://docs.obspy.org/packages/obspy.earthworm.html) - [Earthworm](http://folkworm.ceri.memphis.edu/ew-doc/) request client ("experimental")
* [obspy.iris](http://docs.obspy.org/packages/obspy.iris.html) - [IRIS DMC Core Web services](http://www.iris.edu/ws) request client
* [obspy.neries](http://docs.obspy.org/packages/obspy.neries.html) - [NERIES Seismic Data Portal](http://www.seismicportal.eu/jetspeed/portal/) request client
* [obspy.seishub](http://docs.obspy.org/packages/obspy.seishub.html) - [SeisHub](http://www.seishub.org) database client

### References

1. Moritz Beyreuther, Robert Barsch, Lion Krischer, Tobias Megies, Yannik Behr and Joachim Wassermann (2010),
   [ObsPy: A Python Toolbox for Seismology](http://www.seismosoc.org/publications/SRL/SRL_81/srl_81-3_es/), SRL, 81(3), 530-533.
2. Tobias Megies, Moritz Beyreuther, Robert Barsch, Lion Krischer, Joachim Wassermann (2011),
    [ObsPy – What can it do for data centers and observatories?](http://www.annalsofgeophysics.eu/index.php/annals/article/view/4838) Annals Of Geophysics, 54(1), 47-58, doi:10.4401/ag-4838.

## Developer Corner

 * [Style Guide](http://docs.obspy.org/coding_style.html)
 * Best Practices
 * Performance Tips [Python](http://wiki.python.org/moin/PythonSpeed/PerformanceTips), [NumPy and ctypes](http://www.scipy.org/Cookbook/Ctypes), [SciPy](http://www.scipy.org/PerformancePython),  
   [NumPy Book](http://www.tramy.us/numpybook.pdf)
 * Testing & Debugging, Sphinx Documentation
 * [Docs or doesn't exist!](http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/)
 * BrainStorming
