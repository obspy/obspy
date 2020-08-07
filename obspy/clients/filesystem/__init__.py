# -*- coding: utf-8 -*-
r"""
obspy.clients.filesystem - Local filesystem (SDS or TSINDEX) client for ObsPy
=============================================================================
This package provides read support for some local directory structures.

The SDS :class:`~obspy.clients.filesystem.sds.Client` class provides read
support for the SeisComP Data Structure 'SDS' ordered local directory
structure. The SDS client supports any filetypes readable by one of ObsPy's
I/O plugins.

The TSIndex :class:`~obspy.clients.filesystem.tsindex.Client` class provides
read support for miniSEED files indexed using the IRIS
`mseedindex <https://github.com/iris-edu/mseedindex/>`_ program or
:class:`~obspy.clients.filesystem.tsindex.Indexer` class. The
:class:`~obspy.clients.filesystem.tsindex.Indexer` class provides support for
indexing any arbitrary directory tree structure of miniSEED files
into a SQLite3 database that follows the IRIS `tsindex database
schema <https://github.com/iris-edu/mseedindex/wiki/Database-Schema>`_\. This
SQLite3 database can then be used by the
:class:`~obspy.clients.filesystem.tsindex.Client` for timeseries data
extraction.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
