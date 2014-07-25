obspy_dyna
==========

DYNA format read/write module for ObsPy


Overview
--------

The obspy_dyna package contains methods in order to read and write
seismogram files in the DYNA format as defined by the ITACA team
at INGV Milano.

DYNA 1.0 format specifications are available at the following URL:

http://itaca.mi.ingv.it/static_italy_20/doc/Manual_ITACA_beta_version.pdf

Old ITACA format (deprecated) is supported in read-only mode.



Dependencies
------------
* obspy

In order to install obspy_dyna you need a working ObsPy installation
(version 0.9.2 or above) on your system.

ObsPy is a Python toolbox for seismology. It is available at the following URL:

https://github.com/obspy/obspy



Installation
------------

Installation is straightforward:

unpack the archive, cd in the obspy_dyna directory and type

"python setup.py install" as root



Usage
-----

After installation, DYNA seismogram files can be read in ObsPy as
for any other format, by using the "read" function, as described in
the official ObsPy tutorial, available at the following URL:

http://docs.obspy.org/tutorial/code_snippets/reading_seismograms.html

Saving a DYNA file is just a matter of specifying 'format=DYNA' in
the obspy "write" function.



Copyright
---------
GNU Lesser General Public License, Version 3 (LGPLv3)

Copyright (c) 2012 by:
	* The ITACA Development Team (itaca@mi.ingv.it)
    * Emiliano Russo (emiliano.russo@ingv.it)
    * Rodolfo Puglia (rodolfo.puglia@ingv.it)

