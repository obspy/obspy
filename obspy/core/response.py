# -*- coding: utf-8 -*-

from obspy.core import AttribDict


class PAZ(AttribDict):
    """
    Poles, zeros, normalization factor and sensitivity of a seismometer.

    The seismometer is represented as a dictionary containing the fields:

    :type poles: list of complex
    :param poles: Complex poles of the seismometer
    :type zeros: list of complex
    :param zeros: Complex zeros of the seismometer
    :type normalization_factor: float, optional
    :param normalization_factor: multiplicative factor to normalize the filter.
        Defaults to ``1.0``.
    :type normalization_frequency: float, optional
    :param normalization_frequency: The frequency in Hertz, at which the
        normalization_factor is normalized (if any). Defaults to ``1.0``.
    :type sensitivity: float, optional
    :param sensitivity: Overall sensitivity/gain of seismometer.
    :type name: string, optional
    :param name: Instrument name. Default to empty string.
    :type comments: string, optional
    :param comments: Additional comments. Default to empty string.

    .. seealso:: http://www.iris.edu/manuals/SEEDManual_V2.4.pdf
    .. seealso:: http://seismic-handler.org/portal/wiki/HowTo/SimulationFilters
    """
    zeros = []
    poles = []
    normalization_factor = 1.0
    normalization_frequency = 1.0
    sensitivity = 1.0
    name = ''
    comments = ''
