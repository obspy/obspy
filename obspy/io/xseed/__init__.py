# -*- coding: utf-8 -*-
"""
obspy.io.xseed - (X)SEED and RESP support for ObsPy
===================================================
`XML-SEED` was introduced by Tsuboi, Tromp and Komatitsch (2004), it is a XML
representation of `Dataless SEED`. This module contains converters from
`Dataless SEED` to `XML-SEED` and vice versa as well as a converter from
`Dataless SEED` to `RESP` files. The :mod:`~obspy.io.xseed` module is tested
against the complete ORFEUS Dataless SEED archive, the IRIS (US) Dataless SEED
archive and against ArcLink response requests.

All files can be converted to ObsPy's internal inventory objects at which
point they can be written out to any format ObsPy supports. In the case of
RESP files these are potentially incomplete as RESP files lack vital
information like geographical coordinates.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Allocate a Parser object and read/write
---------------------------------------

>>> from obspy.io.xseed import Parser
>>> sp = Parser("/path/to/dataless.seed.BW_FURT")
>>> sp.write_xseed("dataless.seed.BW_RJOB.xml") #doctest: +SKIP

The lines above will convert `Dataless SEED`, e.g.::

    000001V 010009402.3121970,001,00:00:00.0000~2038,001,00:00:00.0000~
    2009,037,04:32:41.0000~BayernNetz~~0110032002RJOB 000003RJOB 000008
    ...

to the `XML-SEED` representation, e.g.::

    <?xml version='1.0' encoding='utf-8'?>
    <xseed version="1.0">
      <volume_index_control_header>
        <volume_identifier blockette="010">
          <version_of_format>2.4</version_of_format>
          <logical_record_length>12</logical_record_length>
          <beginning_time>1970-01-01T00:00:00</beginning_time>
          <end_time>2038-01-01T00:00:00</end_time>
          <volume_time>2009-02-06T04:32:41</volume_time>
          <originating_organization>BayernNetz</originating_organization>
    ...


A response file can be written in a similar manner, just replace
:meth:`~obspy.io.xseed.parser.Parser.write_xseed` by
:meth:`~obspy.io.xseed.parser.Parser.write_resp`:

>>> sp.write_resp(folder="BW_FURT", zipped=False) #doctest: +SKIP


The Parser Object
-----------------

`SEED` files as well as its derived format `XML-SEED` will be
parsed in a :class:`~obspy.io.xseed.parser.Parser` structure.

`SEED` volumes have four different volume types:

* Volume Index Control Headers
* Abbreviation Dictionary Control Headers
* Station Control Headers
* Time Span Control Headers (currently not supported by ObsPy. Some dummy
  headers will be written in case they are needed by SEED/XSEED conventions.)

After parsing a `SEED` or `XML-SEED` file the Blockette objects for each
volume will be stored in the attributes``Parser.volume``,
``Parser.abbreviations`` and ``Parser.stations``. Each item is a list of all
related Blockettes and ``Parser.stations`` is a list of stations which contains
all related Blockettes.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

# needs to stay above import statements
DEFAULT_XSEED_VERSION = '1.1'


class InvalidResponseError(Exception):
    """
    Raised when a response is clearly invalid.

    The error message should give a clearer idea of why.
    """
    pass


from .parser import Parser


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
