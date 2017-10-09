#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Routing client for the IRIS federator routing service.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
    IRIS-DMC
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .routing_client import BaseRoutingClient


class FederatorRoutingClient(BaseRoutingClient):
    def __init__(self, *args, **kwargs):
        pass

    def get_waveforms(self):


if __name__ == '__main__':  # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
