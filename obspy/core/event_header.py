from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util import Enum


OriginUncertaintyDescription = Enum([
    "horizontal uncertainty",
    "uncertainty ellipse",
    "confidence ellipsoid",
])

AmplitudeCategory = Enum([
    "point",
    "mean",
    "duration",
    "period",
    "integral",
    "other",
])

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

OriginType = Enum([
    "hypocenter",
    "centroid",
    "amplitude",
    "macroseismic",
    "rupture start",
    "rupture end",
])

MTInversionType = Enum([
    "general",
    "zero trace",
    "double couple",
])

EvaluationMode = Enum([
    "manual",
    "automatic",
])

EvaluationStatus = Enum([
    "preliminary",
    "confirmed",
    "reviewed",
    "final",
    "rejected",
])

PickOnset = Enum([
    "emergent",
    "impulsive",
    "questionable",
])

DataUsedWaveType = Enum([
    "P waves",
    "body waves",
    "surface waves",
    "mantle waves",
    "combined",
    "unknown",
])

AmplitudeUnit = Enum([
    "m",
    "s",
    "m/s",
    "m/(s*s)",
    "m*s",
    "dimensionless",
    "other",
])

EventDescriptionType = Enum([
    "felt report",
    "Flinn-Engdahl region",
    "local time",
    "tectonic summary",
    "nearest cities",
    "earthquake name",
    "region name",
])

MomentTensorCategory = Enum([
    "teleseismic",
    "regional",
])

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

EventTypeCertainty = Enum([
    "known",
    "suspected",
])

SourceTimeFunctionType = Enum([
    "box car",
    "triangle",
    "trapezoid",
    "unknown",
])

PickPolarity = Enum([
    "positive",
    "negative",
    "undecidable",
])
