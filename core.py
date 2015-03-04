# -*- coding: utf-8 -*-
"""
Created on Wed 2015-03-04 00:00 UTC

@author: Mike Turnbull

Kelunji Classic bindings to ObsPy core module.
"""

from obspy import Trace, UTCDateTime, Stream
import numpy as np
import warnings

def IsKelunjiClassic(filename):
    """
    Checks whether a file is Kelunji Classic (KC) seismogram or not.

    :type filename: string
    :param filename: Potential KC file to be checked.
    :rtype: integer
    :return: '1' if it is a fully readable KC file. '0' if it is not a KC file. '-1' if it is a partially readable KC file.
    """
    
    #TODO: * Read in sufficient of the header to check ID words.
    #      * Gather enough meta data to cross-check with the trace
    #        data to ensure that it is a fully readable file.

def readKelunjiClassic(filename, **kwargs):  # @UnusedVariable
    """
    Reads a Kelunji Classic seismogram file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :param filename: Kelunji Classic seismogram file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.
    """
    
    #TODO: 

