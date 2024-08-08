# -*- coding: utf-8 -*-
"""

This module provides enumerations defined in the
`QuakeML <https://quake.ethz.ch/quakeml/>`_ standard.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.util import Enum


ATTRIBUTE_HAS_ERRORS = True


AmplitudeCategory = Enum([
    "point",
    "mean",
    "duration",
    "period",
    "integral",
    "other",
])
"""
Amplitude category.  This attribute describes the way the
waveform trace is evaluated to derive an amplitude value. This can be
just reading a single value for a given point in time (point), taking a
mean value over a time interval (mean), integrating the trace over a
time interval (integral), specifying just a time interval (duration),
or evaluating a period (period).
Allowed values are:

* ``"point"``
* ``"mean"``
* ``"duration"``
* ``"period"``
* ``"integral"``
* ``"other"``
"""

AmplitudeUnit = Enum([
    "m",
    "s",
    "m/s",
    "m/(s*s)",
    "m*s",
    "dimensionless",
    "other",
])
"""
Amplitude unit. Values are specified as combinations of SI base units.
Allowed values are:

* ``"m"``
* ``"s"``
* ``"m/s"``
* ``"m/(s*s)"``
* ``"m*s"``
* ``"dimensionless"``
* ``"other"``
"""

DataUsedWaveType = Enum([
    "P waves",
    "body waves",
    "surface waves",
    "mantle waves",
    "combined",
    "unknown",
])
"""
Type of waveform data. Allowed values are:

* ``"P waves"``
* ``"body waves"``
* ``"surface waves"``
* ``"mantle waves"``
* ``"combined"``
* ``"unknown"``
"""

EvaluationMode = Enum([
    "manual",
    "automatic",
])
"""
Evaluation mode. Allowed values are:

* ``"manual"``
* ``"automatic"``
"""

EvaluationStatus = Enum([
    "preliminary",
    "confirmed",
    "reviewed",
    "final",
    "rejected",
])
"""
Evaluation status. Allowed values are:

* ``"preliminary"``
* ``"confirmed"``
* ``"reviewed"``
* ``"final"``
* ``"rejected"``
"""

EventDescriptionType = Enum([
    "felt report",
    "Flinn-Engdahl region",
    "local time",
    "tectonic summary",
    "nearest cities",
    "earthquake name",
    "region name",
])
"""
Category of earthquake description. Allowed values are:

* ``"felt report"``
* ``"Flinn-Engdahl region"``
* ``"local time"``
* ``"tectonic summary"``
* ``"nearest cities"``
* ``"earthquake name"``
* ``"region name"``
"""

EventType = Enum([
    "not existing",
    "not reported",
    "earthquake",
    "anthropogenic event",
    "collapse",
    "cavity collapse",
    "mine collapse",
    "building collapse",
    "explosion",
    "accidental explosion",
    "chemical explosion",
    "controlled explosion",
    "experimental explosion",
    "industrial explosion",
    "mining explosion",
    "quarry blast",
    "road cut",
    "blasting levee",
    "nuclear explosion",
    "induced or triggered event",
    "rock burst",
    "reservoir loading",
    "fluid injection",
    "fluid extraction",
    "crash",
    "plane crash",
    "train crash",
    "boat crash",
    "other event",
    "atmospheric event",
    "sonic boom",
    "sonic blast",
    "acoustic noise",
    "thunder",
    "avalanche",
    "snow avalanche",
    "debris avalanche",
    "hydroacoustic event",
    "ice quake",
    "slide",
    "landslide",
    "rockslide",
    "meteorite",
    "volcanic eruption",
], replace={'other': 'other event'})
"""
Describes the type of an event. Allowed values are:

* ``"not existing"``
* ``"not reported"``
* ``"earthquake"``
* ``"anthropogenic event"``
* ``"collapse"``
* ``"cavity collapse"``
* ``"mine collapse"``
* ``"building collapse"``
* ``"explosion"``
* ``"accidental explosion"``
* ``"chemical explosion"``
* ``"controlled explosion"``
* ``"experimental explosion"``
* ``"industrial explosion"``
* ``"mining explosion"``
* ``"quarry blast"``
* ``"road cut"``
* ``"blasting levee"``
* ``"nuclear explosion"``
* ``"induced or triggered event"``
* ``"rock burst"``
* ``"reservoir loading"``
* ``"fluid injection"``
* ``"fluid extraction"``
* ``"crash"``
* ``"plane crash"``
* ``"train crash"``
* ``"boat crash"``
* ``"other event"``
* ``"atmospheric event"``
* ``"sonic boom"``
* ``"sonic blast"``
* ``"acoustic noise"``
* ``"thunder"``
* ``"avalanche"``
* ``"snow avalanche"``
* ``"debris avalanche"``
* ``"hydroacoustic event"``
* ``"ice quake"``
* ``"slide"``
* ``"landslide"``
* ``"rockslide"``
* ``"meteorite"``
* ``"volcanic eruption"``
"""

EventTypeCertainty = Enum([
    "known",
    "suspected",
])
"""
Denotes how certain the information on event type is. Allowed values are:

* ``"suspected"``
* ``"known"``
"""

MTInversionType = Enum([
    "general",
    "zero trace",
    "double couple",
])
"""
Moment tensor inversion type. Allowed values are:

* ``"general"``
* ``"zero trace"``
* ``"double couple"``
"""

MomentTensorCategory = Enum([
    "teleseismic",
    "regional",
])
"""
Moment tensor category. Allowed values are:

* ``"teleseismic"``
* ``"regional"``
"""

OriginDepthType = Enum([
    "from location",
    "from moment tensor inversion",
    "from modeling of broad-band P waveforms",
    "constrained by depth phases",
    "constrained by direct phases",
    "constrained by depth and direct phases",
    "operator assigned",
    "other",
])
"""
Type of origin depth determination. Allowed values are:

* ``"from location"``
* ``"from moment tensor inversion"``
* ``"from modeling of broad-band P waveforms"``
* ``"constrained by depth phases"``
* ``"constrained by direct phases"``
* ``"constrained by depth and direct phases"``
* ``"operator assigned"``
* ``"other"``
"""

OriginType = Enum([
    "hypocenter",
    "centroid",
    "amplitude",
    "macroseismic",
    "rupture start",
    "rupture end",
])
"""
Origin type. Allowed values are:

* ``"hypocenter"``
* ``"centroid"``
* ``"amplitude"``
* ``"macroseismic"``
* ``"rupture start"``
* ``"rupture end"``
"""

OriginUncertaintyDescription = Enum([
    "horizontal uncertainty",
    "uncertainty ellipse",
    "confidence ellipsoid",
])
"""
Preferred origin uncertainty description. Allowed values are:

* ``"horizontal uncertainty"``
* ``"uncertainty ellipse"``
* ``"confidence ellipsoid"``
"""

PickOnset = Enum([
    "emergent",
    "impulsive",
    "questionable",
])
"""
Flag that roughly categorizes the sharpness of the pick onset.
Allowed values are:

* ``"emergent"``
* ``"impulsive"``
* ``"questionable"``
"""

PickPolarity = Enum([
    "positive",
    "negative",
    "undecidable",
])
"""
Indicates the polarity of first motion, usually from impulsive onsets.
Allowed values are:

* ``"positive"``
* ``"negative"``
* ``"undecidable"``
"""

SourceTimeFunctionType = Enum([
    "box car",
    "triangle",
    "trapezoid",
    "unknown",
])
"""
Type of source time function. Allowed values are:

* ``"box car"``
* ``"triangle"``
* ``"trapezoid"``
* ``"unknown"``
"""
