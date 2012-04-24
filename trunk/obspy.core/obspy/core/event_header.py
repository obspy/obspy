from obspy.core.util import Enum

OriginUncertaintyDescription = Enum([
    "horizontal uncertainty",
    "uncertainty ellipse",
    "confidence ellipsoid",
    "probability density function",
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
    "earthquake",
    "induced earthquake",
    "quarry blast",
    "explosion",
    "chemical explosion",
    "nuclear explosion",
    "landslide",
    "rockslide",
    "snow avalanche",
    "debris avalanche",
    "mine collapse",
    "building collapse",
    "volcanic eruption",
    "meteor impact",
    "plane crash",
    "sonic boom",
    "not existing",
    "other",
    "null",
])
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
