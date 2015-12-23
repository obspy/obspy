# -*- coding: utf-8 -*-
"""
obspy.core.event - Classes for handling event metadata
======================================================
This module provides a class hierarchy to consistently handle event metadata.
This class hierarchy is closely modelled after the de-facto standard
format `QuakeML <https://quake.ethz.ch/quakeml/>`_.

.. figure:: /_images/Event.png

.. note::

    For handling additional information not covered by the QuakeML standard and
    how to output it to QuakeML see the :ref:`ObsPy Tutorial <quakeml-extra>`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future.builtins import *  # NOQA

from .base import *
from .event import Event
from .catalog import Catalog
from .radpattern import plot_3drpattern, farfield_p, farfield_s

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
