# -*- coding: utf-8 -*-
"""
obspy.io.json - JSON write support for ObsPy
============================================

JavaScript Object Notation is a text-based open standard designed for
human-readable data interchange. The JSON format is often used for serializing
and transmitting structured data over a network connection. It is used
primarily to transmit data between a server and web application, serving as an
alternative to XML.

See the module :mod:`obspy.io.json.default` for documentation on the class.
A write function for files and a utility for compact string serialization using
the Default class are located in :mod:`obspy.io.json.core`.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .default import Default
from .core import get_dump_kwargs, _write_json
