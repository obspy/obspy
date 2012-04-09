# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Catalog and Event objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile, getExampleFile, Enum, \
    uncompressFile, AttribDict, _readFromPlugin
import copy
import glob
import os
import urllib2


def readEvents(pathname_or_url=None):
    """
    Read event files into an ObsPy Catalog object.

    The :func:`~obspy.core.event.readEvents` function opens either one or
    multiple event files given via file name or URL using the
    ``pathname_or_url`` attribute.

    :type pathname_or_url: string, optional
    :param pathname_or_url: String containing a file name or a URL. Wildcards
        are allowed for a file name. If this attribute is omitted, a Catalog
        object with an example data set will be created.
    :type format: string, optional
    :return: A ObsPy :class:`~obspy.core.event.Catalog` object.
    """
    # if no pathname or URL specified, make example stream
    if not pathname_or_url:
        return _createExampleCatalog()
    # if pathname starts with /path/to/ try to search in examples
    if isinstance(pathname_or_url, basestring) and \
       pathname_or_url.startswith('/path/to/'):
        try:
            pathname_or_url = getExampleFile(pathname_or_url[9:])
        except:
            # otherwise just try to read the given /path/to folder
            pass
    # create catalog
    cat = Catalog()
    if "://" in pathname_or_url:
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        # some URL
        fh = NamedTemporaryFile(suffix=suffix)
        fh.write(urllib2.urlopen(pathname_or_url).read())
        fh.close()
        cat.extend(_read(fh.name).events)
        os.remove(fh.name)
    else:
        # file name
        pathname = pathname_or_url
        for file in glob.iglob(pathname):
            cat.extend(_read(file).events)
        if len(cat) == 0:
            # try to give more specific information why the stream is empty
            if glob.has_magic(pathname) and not glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not glob.has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)
    return cat


@uncompressFile
def _read(filename, format=None, **kwargs):
    """
    Reads a single event file into a ObsPy Catalog object.
    """
    catalog, format = _readFromPlugin('event', filename, format=format,
                                      **kwargs)
    for event in catalog:
        event._format = format
    return catalog


def _createExampleCatalog():
    """
    Create an example catalog.
    """
    return readEvents('/path/to/neries_events.xml')


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


class CreationInfo(AttribDict):
    """
    CreationInfo is used to describe author, version, and creation time of a
    resource.

    :type agency_id: str, optional
    :param agency_id: Designation of agency that published a resource.
    :type agency_uri: str, optional
    :param agency_uri: Resource identifier of the agency that published a
        resource.
    :type author: str, optional
    :param author: Name describing the author of a resource.
    :type author_uri: str, optional
    :param author_uri: Resource identifier of the author of a resource.
    :type creation_time: UTCDateTime, optional
    :param creation_time: Time of creation of a resource.
    :type version: str, optional
    :param version: Version string of a resource.
    """
    agency_id = None
    agency_uri = None
    author = None
    author_uri = None
    creation_time = None
    version = None


class _ValueQuantity(AttribDict):
    """
    Physical quantities that can be expressed numerically — either as integers,
    floating point numbers or UTCDateTime objects — are represented by their
    measured or computed values and optional values for symmetric or upper and
    lower uncertainties.

    :type value: int, float or :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param value: Value of the quantity. The unit is implicitly defined and
        depends on the context.
    :type uncertainty: float, optional
    :param uncertainty: Symmetric uncertainty or boundary.
    :type lower_uncertainty: float, optional
    :param lower_uncertainty: Relative lower uncertainty or boundary.
    :type upper_uncertainty: float, optional
    :param upper_uncertainty: Relative upper uncertainty or boundary.
    :type confidence_level: float, optional
    :param confidence_level: Confidence level of the uncertainty, given in
        percent.
    """
    value = None
    uncertainty = None
    lower_uncertainty = None
    upper_uncertainty = None
    confidence_level = None


class TimeQuantity(_ValueQuantity):
    value_type = UTCDateTime


class FloatQuantity(_ValueQuantity):
    value_type = float


class IntegerQuantity(_ValueQuantity):
    value_type = int


class CompositeTime(AttribDict):
    """
    Focal times differ significantly in their precision. While focal times of
    instrumentally located earthquakes are estimated precisely down to seconds,
    historic events have only incomplete time descriptions. Sometimes, even
    contradictory information about the rupture time exist. The CompositeTime
    type allows for such complex descriptions.

    :type year: :class:`~obspy.core.event.IntegerQuantity`
    :param year: Year or range of years of the event’s focal time.
    :type month: :class:`~obspy.core.event.IntegerQuantity`
    :param month: Month or range of months of the event’s focal time.
    :type day: :class:`~obspy.core.event.IntegerQuantity`
    :param day: Day or range of days of the event’s focal time.
    :type hour: :class:`~obspy.core.event.IntegerQuantity`
    :param hour: Hour or range of hours of the event’s focal time.
    :type minute: :class:`~obspy.core.event.IntegerQuantity`
    :param minute: Minute or range of minutes of the event’s focal time.
    :type second: :class:`~obspy.core.event.FloatQuantity`
    :param second: Second and fraction of seconds or range of seconds with
        fraction of the event’s focal time.
    """
    year = IntegerQuantity()
    month = IntegerQuantity()
    day = IntegerQuantity()
    hour = IntegerQuantity()
    minute = IntegerQuantity()
    second = FloatQuantity()


class Comment(AttribDict):
    """
    Comment holds information on comments to a resource as well as author and
    creation time information.

    :type text: str
    :param text: Text of comment.
    :type id: str or None, optional
    :param id: Identifier of comment, in QuakeML resource identifier format.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`
    :param creation_info: Creation info of comment (author, version, creation
        time).
    """
    text = ''
    id = None
    creation_info = CreationInfo()


class OriginQuality(AttribDict):
    """
    This class contains various attributes commonly used to describe the
    quality of an origin, e. g., errors, azimuthal coverage, etc.

    :type associated_phase_count: int, optional
    :param associated_phase_count: Number of associated phases, regardless of
        their use for origin computation.
    :type used_phase_count: int, optional
    :param used_phase_count: Number of defining phases, i.e., phase
        observations that were actually used for computing the origin. Note
        that there may be more than one defining phase per station.
    :type associated_station_count: int, optional
    :param associated_station_count: Number of stations at which the event was
        observed.
    :type used_station_count: int, optional
    :param used_station_count: Number of stations from which data was used for
        origin computation.
    :type depth_phase_count: int, optional
    :param depth_phase_count: Number of depth phases (typically pP, sometimes
        sP) used in depth computation.
    :type standard_error: float, optional
    :param standard_error: RMS of the travel time residuals of the arrivals
        used for the origin computation. Unit: s
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Largest azimuthal gap in station distribution as seen
        from epicenter. Unit: deg
    :type secondary_azimuthal_gap: float, optional
    :param secondary_azimuthal_gap: Secondary azimuthal gap in station
        distribution, i. e., the largest azimuthal gap a station closes.
        Unit: deg
    :type ground_truth_level: str, optional
    :param ground_truth_level: String describing ground-truth level, e. g. GT0,
        GT5, etc.
    :type minimum_distance: float, optional
    :param minimum_distance: Distance Epicentral distance of station closest to
        the epicenter. Unit: deg
    :type maximum_distance: float, optional
    :param maximum_distance: Distance Epicentral distance of station farthest
        from the epicenter. Unit: deg
    :type median_distance: float, optional
    :param median_distance: Distance Median epicentral distance of used
        stations. Unit: deg
    """
    associated_phase_count = None
    used_phase_count = None
    associated_station_count = None
    used_station_count = None
    depth_phase_count = None
    standard_error = None
    azimuthal_gap = None
    secondary_azimuthal_gap = None
    ground_truth_level = None
    minimum_distance = None
    maximum_distance = None
    median_distance = None


class ConfidenceEllipsoid(AttribDict):
    """
    This class represents a description of the location uncertainty as a
    confidence ellipsoid with arbitrary orientation in space.

    :param semi_major_axis_length: Largest uncertainty, corresponding to the
        semi-major axis of the confidence ellipsoid. Unit: m
    :param semi_minor_axis_length: Smallest uncertainty, corresponding to the
        semi-minor axis of the confidence ellipsoid. Unit: m
    :param semi_intermediate_axis_length: Uncertainty in direction orthogonal
        to major and minor axes of the confidence ellipsoid. Unit: m
    :param major_axis_plunge: Plunge angle of major axis of confidence
        ellipsoid. Unit: deg
    :param major_axis_azimuth: Azimuth angle of major axis of confidence
        ellipsoid. Unit: deg
    :param major_axis_rotation: This angle describes a rotation about the
        confidence ellipsoid’s major axis which is required to define the
        direction of the ellipsoid’s minor axis. A zero majorAxisRotation angle
        means that the minor axis lies in the plane spanned by the major axis
        and the vertical. Unit: deg
    """
    semi_major_axis_length = None
    semi_minor_axis_length = None
    semi_intermediate_axis_length = None
    major_axis_plunge = None
    major_axis_azimuth = None
    major_axis_rotation = None


class OriginUncertainty(AttribDict):
    """
    This class describes the location uncertainties of an origin.

    The uncertainty can be described either as a simple circular horizontal
    uncertainty, an uncertainty ellipse according to IMS1.0, or a confidence
    ellipsoid. The preferred variant can be given in the attribute
    ``preferred_description``.

    :type preferred_description: str, optional
    :param preferred_description: Preferred uncertainty description. Allowed
        values are the following::
            * horizontal uncertainty
            * uncertainty ellipse
            * confidence ellipsoid
            * probability density function
    :type horizontal_uncertainty: float, optional
    :param horizontal_uncertainty: Circular confidence region, given by single
        value of horizontal uncertainty. Unit: m
    :type min_horizontal_uncertainty: float, optional
    :param min_horizontal_uncertainty: Semi-major axis of confidence ellipse.
        Unit: m
    :type max_horizontal_uncertainty: float, optional
    :param max_horizontal_uncertainty: Semi-minor axis of confidence ellipse.
        Unit: m
    :type azimuth_max_horizontal_uncertainty: float, optional
    :param azimuth_max_horizontal_uncertainty: Azimuth of major axis of
        confidence ellipse. Unit: deg
    :type confidence_ellipsoid: :class:`~obspy.core.event.ConfidenceEllipsoid`,
        optional
    :param confidence_ellipsoid: Confidence ellipsoid
    """
    horizontal_uncertainty = None
    min_horizontal_uncertainty = None
    max_horizontal_uncertainty = None
    azimuth_max_horizontal_uncertainty = None
    confidence_ellipsoid = ConfidenceEllipsoid()

    def _getOriginUncertaintyDescription(self):
        return self.__dict__.get('preferred_description', None)

    def _setOriginUncertaintyDescription(self, value):
        self.__dict__['preferred_description'] = \
            OriginUncertaintyDescription(value)

    preferred_description = property(_getOriginUncertaintyDescription,
                                     _setOriginUncertaintyDescription)


class Origin(object):
    """
    This class represents the focal time and geographical location of an
    earthquake hypocenter, as well as additional meta-information.

    :type public_id: str
    :param public_id: Resource identifier of Origin.
    :type time: :class:`~obspy.core.event.TimeQuantity`
    :param time: Focal time.
    :type latitude: :class:`~obspy.core.event.FloatQuantity`
    :param latitude: Hypocenter latitude. Unit: deg
    :type longitude: :class:`~obspy.core.event.FloatQuantity`
    :param longitude: Hypocenter longitude. Unit: deg
    :type depth: :class:`~obspy.core.event.FloatQuantity`, optional
    :param depth: Depth of hypocenter. Unit: m
    :type depth_type: str, optional
    :param depth_type: Type of depth determination. Allowed values are the
        following:
            * ``"from location"``
            * ``"constrained by depth phases"``
            * ``"constrained by direct phases"``
            * ``"operator assigned"``
            * ``"other"``
    :type time_fixed: bool, optional
    :param time_fixed: ``True`` if focal time was kept fixed for computation
        of the Origin.
    :type epicenter_fixed: bool, optional
    :param epicenter_fixed: ``True`` if epicenter was kept fixed for
        computation of Origin.
    :type reference_system_id: str, optional
    :param reference_system_id: Identifies the reference system used for
        hypocenter determination.
    :type method_id: str, optional
    :param method_id: Identifies the method used for locating the event.
    :type earth_model_id: str, optional
    :param earth_model_id: Identifies the earth model used in ``method_id``.
    :type composite_times: list of :class:`~obspy.core.event.CompositeTime`,
        optional
    :param composite_times: Supplementary information on time of rupture start.
        Complex descriptions of focal times of historic event are possible,
        see description of the :class:`~obspy.core.event.CompositeTime` class.
    :type quality: :class:`~obspy.core.event.OriginQuality`, optional
    :param quality: Additional parameters describing the quality of an origin
        determination.
    :type type: str, optional
    :param type: Describes the origin type. Allowed values are the
        following:
            * ``"rupture start"``
            * ``"centroid"``
            * ``"rupture end"``
            * ``"hypocenter"``
            * ``"amplitude"``
            * ``"macroseismic"``
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Origin. Allowed values are the
        following:
            * ``"manual"``
            * ``"automatic"``
    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Origin. Allowed values are
        the following:
            * ``"preliminary"``
            * ``"confirmed"``
            * ``"reviewed"``
            * ``"final"``
            * ``"rejected"``
            * ``"reported"``
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    # QuakeML attributes
    public_id = ''
    time = TimeQuantity()
    latitude = FloatQuantity()
    longitude = FloatQuantity()
    depth = FloatQuantity()
    time_fixed = None
    epicenter_fixed = None
    reference_system_id = None
    method_id = None
    earth_model_id = None
    composite_times = []
    quality = OriginQuality()
    origin_uncertainty = OriginUncertainty()
    comments = []
    creation_info = CreationInfo()
    # child elements
    arrivals = []

    def __str__(self):
        return self._pretty_str(['time', 'latitude', 'longitude'])

    def _getOriginDepthType(self):
        return self.__dict__.get('depth_type', None)

    def _setOriginDepthType(self, value):
        self.__dict__['depth_type'] = OriginDepthType(value)

    depth_type = property(_getOriginDepthType, _setOriginDepthType)

    def _getOriginType(self):
        return self.__dict__.get('type', None)

    def _setOriginType(self, value):
        self.__dict__['type'] = OriginType(value)

    type = property(_getOriginType, _setOriginType)

    def _getEvaluationMode(self):
        return self.__dict__.get('evaluation_mode', None)

    def _setEvaluationMode(self, value):
        self.__dict__['evaluation_mode'] = EvaluationMode(value)

    evaluation_mode = property(_getEvaluationMode, _setEvaluationMode)

    def _getEvaluationStatus(self):
        return self.__dict__.get('evaluation_status', None)

    def _setEvaluationStatus(self, value):
        self.__dict__['evaluation_status'] = EvaluationStatus(value)

    evaluation_status = property(_getEvaluationStatus, _setEvaluationStatus)


class Magnitude(AttribDict):
    """
    Describes a magnitude which can, but need not be associated with an Origin.

    Association with an origin is expressed with the optional attribute
    ``origin_id``. It is either a combination of different magnitude
    estimations, or it represents the reported magnitude for the given Event.

    :type public_id: str
    :param public_id: Resource identifier of Magnitude.
    :type mag: float
    :param mag: Resulting magnitude value from combining values of type
        :class:`~obspy.core.event.StationMagnitude`. If no estimations are
        available, this value can represent the reported magnitude.
    :type type: str, optional
    :param type: Describes the type of magnitude. This is a free-text field
        because it is impossible to cover all existing magnitude type
        designations with an enumeration. Possible values are
            * unspecified magitude (``'M'``),
            * local magnitude (``'ML'``),
            * body wave magnitude (``'Mb'``),
            * surface wave magnitude (``'MS'``),
            * moment magnitude (``'Mw'``),
            * duration magnitude (``'Md'``)
            * coda magnitude (``'Mc'``)
            * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.
    :type origin_id: str, optional
    :param origin_id: Reference to an origin’s public_id if the magnitude has
        an associated Origin.
    :type method_id: str, optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and type.
    :type station_count, int, optional
    :param station_count Number of used stations for this magnitude
        computation.
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Azimuthal gap for this magnitude computation.
        Unit: deg
    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Magnitude. Allowed values
        are the following:
            * ``"preliminary"``
            * ``"confirmed"``
            * ``"reviewed"``
            * ``"final"``
            * ``"rejected"``
            * ``"reported"``
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    public_id = ''
    mag = FloatQuantity()
    type = None
    origin_id = None
    method_id = None
    station_count = None
    azimuthal_gap = None
    comments = []
    creation_info = CreationInfo()

    def __str__(self):
        return self._pretty_str(['magnitude'])

    def _getEvaluationStatus(self):
        return self.__dict__.get('evaluation_status', None)

    def _setEvaluationStatus(self, value):
        self.__dict__['evaluation_status'] = EvaluationStatus(value)

    evaluation_status = property(_getEvaluationStatus, _setEvaluationStatus)


class EventDescription(AttribDict):
    """
    Free-form string with additional event description. This can be a
    well-known name, like 1906 San Francisco Earthquake. A number of categories
    can be given in type.

    :type text: str
    :param text: Free-form text with earthquake description.
    :type type: str, optional
    :param type: Category of earthquake description. Values can be taken from
        the following:
            * ``"felt report"``
            * ``"Flinn-Engdahl region"``
            * ``"local time"``
            * ``"tectonic summary"``
            * ``"nearest cities"``
            * ``"earthquake name"``
            * ``"region name"``
    """
    text = ''
    type = None

    def _getEventDescriptionType(self):
        return self.__dict__.get('type', None)

    def _setEventDescriptionType(self, value):
        self.__dict__['type'] = EventDescriptionType(value)

    type = property(_getEventDescriptionType, _setEventDescriptionType)


class Event(object):
    """
    The class Event describes a seismic event which does not necessarily need
    to be a tectonic earthquake. An event is usually associated with one or
    more origins, which contain information about focal time and geographical
    location of the event. Multiple origins can cover automatic and manual
    locations, a set of location from different agencies, locations generated
    with different location programs and earth models, etc. Furthermore, an
    event is usually associated with one or more magnitudes, and with one or
    more focal mechanism determinations.

    :type public_id: str, optional
    :param public_id: Resource identifier of Event.
    :type preferred_origin_id: str, optional
    :param preferred_origin_id: Refers to the ``public_id`` of the preferred
        :class:`~obspy.core.event.Origin` object.
    :type preferred_magnitude_id: str, optional
    :param preferred_magnitude_id: Refers to the ``public_id`` of the preferred
        :class:`~obspy.core.event.Magnitude` object.
    :type preferred_focal_mechanism_id: str, optional
    :param preferred_focal_mechanism_id: Refers to the ``public_id`` of the
        preferred :class:`~obspy.core.event.FocalMechanism` object.
    :type type: str, optional
    :param type: Describes the type of an event. Allowed values are the
        following:
            * ``"earthquake"``
            * ``"induced earthquake"``
            * ``"quarry blast"``
            * ``"explosion"``
            * ``"chemical explosion"``
            * ``"nuclear explosion"``
            * ``"landslide"``
            * ``"rockslide"``
            * ``"snow avalanche"``
            * ``"debris avalanche"``
            * ``"mine collapse"``
            * ``"building collapse"``
            * ``"volcanic eruption"``
            * ``"meteor impact"``
            * ``"plane crash"``
            * ``"sonic boom"``
            * ``"not existing"``
            * ``"null"``
            * ``"other"``
    :type type_certainty: str, optional
    :param type_certainty: Denotes how certain the information on event type
        is. Allowed values are the following:
            * ``"suspected"``
            * ``"known"``
    :type description: list of :class:`~obspy.core.event.EventDescription`
    :param description: Additional event description, like earthquake name,
        Flinn-Engdahl region, etc.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    # QuakeML attributes
    public_id = ''
    preferred_origin_id = None
    preferred_magnitude_id = None
    preferred_focal_mechanism_id = None
    __type = None
    __type_certainty = None
    descriptions = []
    comments = []
    creation_info = CreationInfo()
    # child elements
    origins = []
    magnitudes = []
    station_magnitudes = []
    focal_mechanism = []
    picks = []
    amplitudes = []

    def __init__(self, public_id='', preferred_origin_id=None,
                 preferred_magnitude_id=None,
                 preferred_focal_mechanism_id=None, type=None,
                 type_certainty=None, descriptions=None, comments=None,
                 creation_info=CreationInfo()):
        self.public_id = public_id
        self.preferred_origin_id = preferred_origin_id
        self.preferred_magnitude_id = preferred_magnitude_id
        self.preferred_focal_mechanism_id = preferred_focal_mechanism_id
        self.type = type
        self.type_certainty = type_certainty
        if descriptions is not None:
            self.descriptions = list(descriptions)
        if comments is not None:
            self.comments = list(comments)
        self.creation_info = creation_info

    def __eq__(self, other):
        """
        Implements rich comparison of Event objects for "==" operator.

        Events are the same, if the have the same id.
        """
        # check if other object is a Event
        if not isinstance(other, Event):
            return False
        if self.id != other.id:
            return False
        return True

    def __str__(self):
        out = ''
        if self.preferred_origin:
            out += '%s | %+7.3f, %+8.3f' % (self.preferred_origin.time.value,
                                       self.preferred_origin.latitude.value,
                                       self.preferred_origin.longitude.value)
        if self.preferred_magnitude:
            out += ' | %s %-2s' % (self.preferred_magnitude.mag.value,
                                   self.preferred_magnitude.type)
        if self.preferred_origin and self.preferred_origin.evaluation_mode:
            out += ' | %s' % (self.preferred_origin.evaluation_mode)
        return out

    def _getEventType(self):
        return self.__type

    def _setEventType(self, value):
        self.__type = EventType(value)

    type = property(_getEventType, _setEventType)

    def _getEventTypeCertainty(self):
        return self.__type_certainty

    def _setEventTypeCertainty(self, value):
        self.__type_certainty = EventTypeCertainty(value)

    type_certainty = property(_getEventTypeCertainty, _setEventTypeCertainty)

    def _getPreferredMagnitude(self):
        if self.magnitudes:
            return self.magnitudes[0]
        return None

    preferred_magnitude = property(_getPreferredMagnitude)

    def _getPreferredOrigin(self):
        if self.origins:
            return self.origins[0]
        return None

    preferred_origin = property(_getPreferredOrigin)

    def _getPreferredFocalMechanism(self):
        if self.focal_mechanism:
            return self.focal_mechanism[0]
        return None

    preferred_focal_mechanism = property(_getPreferredFocalMechanism)

    def _getTime(self):
        return self.preferred_origin.time

    time = datetime = property(_getTime)

    def _getLatitude(self):
        return self.preferred_origin.latitude

    latitude = lat = property(_getLatitude)

    def _getLongitude(self):
        return self.preferred_origin.longitude

    longitude = lon = property(_getLongitude)

    def _getMagnitude(self):
        return self.preferred_magnitude.magnitude

    magnitude = mag = property(_getMagnitude)

    def _getMagnitudeType(self):
        return self.preferred_magnitude.type

    magnitude_type = mag_type = property(_getMagnitudeType)

    def getId(self):
        """
        Returns the identifier of the event.

        :rtype: str
        :return: event identifier
        """
        if self.public_id is None:
            return ''
        return "%s" % (self.public_id)

    id = property(getId)


class Catalog(object):
    """
    This class serves as a container for Event objects.

    :type events: list of :class:`~obspy.core.event.Event`, optional
    :param events: List of events
    :type public_id: str, optional
    :param public_id: Resource identifier of the catalog.
    :type description: str, optional
    :param description: Description string that can be assigned to the
        earthquake catalog, or collection of events.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    # QuakeML attributes
    public_id = ''
    description = None
    comments = []
    creation_info = CreationInfo()
    # child elements
    events = []

    def __init__(self, events=[], public_id='', description=None,
                 comments=[], creation_info=CreationInfo()):
        """
        Initializes a Catalog object.
        """
        self.events = []
        if isinstance(events, Event):
            events = [events]
        if events:
            self.events.extend(events)
        self.public_id = public_id
        self.description = description
        self.comments = comments
        self.creation_info = creation_info

    def __add__(self, other):
        """
        Method to add two catalogs.
        """
        if isinstance(other, Event):
            other = Catalog([other])
        if not isinstance(other, Catalog):
            raise TypeError
        events = self.events + other.events
        return self.__class__(events=events)

    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of traces.
        """
        return self.events.__delitem__(index)

    def __eq__(self, other):
        """
        __eq__ method of the Catalog object.

        :type other: :class:`~obspy.core.event.Catalog`
        :param other: Catalog object for comparison.
        :rtype: bool
        :return: ``True`` if both Catalogs contain the same events.

        .. rubric:: Example

        >>> from obspy.core.event import readEvents
        >>> cat = readEvents()
        >>> cat2 = cat.copy()
        >>> cat is cat2
        False
        >>> cat == cat2
        True
        """
        if not isinstance(other, Catalog):
            return False
        if self.events != other.events:
            return False
        return True

    def __getitem__(self, index):
        """
        __getitem__ method of the Catalog object.

        :return: Event objects
        """
        if isinstance(index, slice):
            return self.__class__(events=self.events.__getitem__(index))
        else:
            return self.events.__getitem__(index)

    def __getslice__(self, i, j, k=1):
        """
        __getslice__ method of the Catalog object.

        :return: Catalog object
        """
        # see also http://docs.python.org/reference/datamodel.html
        return self.__class__(events=self.events[max(0, i):max(0, j):k])

    def __iadd__(self, other):
        """
        Method to add two catalog with self += other.

        It will extend the current Catalog object with the events of the given
        Catalog. Events will not be copied but references to the original
        events will be appended.

        :type other: :class:`~obspy.core.event.Catalog` or
            :class:`~obspy.core.event.Event`
        :param other: Catalog or Event object to add.
        """
        if isinstance(other, Event):
            other = Catalog([other])
        if not isinstance(other, Catalog):
            raise TypeError
        self.extend(other.events)
        return self

    def __iter__(self):
        """
        Return a robust iterator for Events of current Catalog.

        Doing this it is safe to remove events from catalogs inside of
        for-loops using catalog's :meth:`~obspy.core.event.Catalog.remove`
        method. Actually this creates a new iterator every time a event is
        removed inside the for-loop.
        """
        return list(self.events).__iter__()

    def __len__(self):
        """
        Returns the number of Events in the Catalog object.
        """
        return len(self.events)

    count = __len__

    def __setitem__(self, index, event):
        """
        __setitem__ method of the Catalog object.
        """
        self.events.__setitem__(index, event)

    def __str__(self):
        """
        Returns short summary string of the current catalog.

        It will contain the number of Events in the Catalog and the return
        value of each Event's :meth:`~obspy.core.event.Event.__str__` method.
        """
        out = str(len(self.events)) + ' Event(s) in Catalog:\n'
        out = out + "\n".join([ev.__str__() for ev in self])
        return out

    def append(self, event):
        """
        Appends a single Event object to the current Catalog object.
        """
        if isinstance(event, Event):
            self.events.append(event)
        else:
            msg = 'Append only supports a single Event object as an argument.'
            raise TypeError(msg)

    def clear(self):
        """
        Clears event list (convenient method).

        .. rubric:: Example

        >>> from obspy.core.event import readEvents
        >>> cat = readEvents()
        >>> len(cat)
        3
        >>> cat.clear()
        >>> cat.events
        []
        """
        self.events = []

    def copy(self):
        """
        Returns a deepcopy of the Catalog object.

        :rtype: :class:`~obspy.core.stream.Catalog`
        :return: Copy of current catalog.

        .. rubric:: Examples

        1. Create a Catalog and copy it

            >>> from obspy.core.event import readEvents
            >>> cat = readEvents()
            >>> cat2 = cat.copy()

           The two objects are not the same:

            >>> cat is cat2
            False

           But they have equal data:

            >>> cat == cat2
            True

        2. The following example shows how to make an alias but not copy the
           data. Any changes on ``st3`` would also change the contents of
           ``st``.

            >>> cat3 = cat
            >>> cat is cat3
            True
            >>> cat == cat3
            True
        """
        return copy.deepcopy(self)

    def extend(self, event_list):
        """
        Extends the current Catalog object with a list of Event objects.
        """
        if isinstance(event_list, list):
            for _i in event_list:
                # Make sure each item in the list is a event.
                if not isinstance(_i, Event):
                    msg = 'Extend only accepts a list of Event objects.'
                    raise TypeError(msg)
            self.events.extend(event_list)
        elif isinstance(event_list, Catalog):
            self.events.extend(event_list.events)
        else:
            msg = 'Extend only supports a list of Event objects as argument.'
            raise TypeError(msg)

    def write(self, filename, format):
        """
        Exports catalog to file system using given format.
        """
        raise NotImplementedError

    def plot(self, resolution='l', **kwargs):  # @UnusedVariable
        """
        Creates preview map of all events in current Catalog object.
        """
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        map = Basemap(resolution=resolution)
        # draw coast lines, country boundaries, fill continents.
        map.drawcoastlines()
        map.drawcountries()
        map.fillcontinents(color='0.8')
        # draw the edge of the map projection region (the projection limb)
        map.drawmapboundary()
        # lat/lon coordinates
        lats = []
        lons = []
        labels = []
        for i, event in enumerate(self.events):
            lats.append(event.preferred_origin.latitude)
            lons.append(event.preferred_origin.longitude)
            labels.append(' #%d' % i)
        # compute the native map projection coordinates for events.
        x, y = map(lons, lats)
        # plot filled circles at the locations of the events.
        map.plot(x, y, 'ro')
        # plot labels
        for name, xpt, ypt in zip(labels, x, y):
            plt.text(xpt, ypt, name, size='small')
        plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
