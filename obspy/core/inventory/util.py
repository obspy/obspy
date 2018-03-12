#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility objects.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy
import re
from numpy import linspace
from matplotlib import transforms as tf
from matplotlib.pyplot import cm
from textwrap import TextWrapper

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import ComparingObject
from obspy.core.util.obspy_types import (FloatWithUncertaintiesAndUnit,
                                         FloatWithUncertaintiesFixedUnit)
from obspy.imaging.util import _set_xaxis_obspy_dates


class BaseNode(ComparingObject):
    """
    From the StationXML definition:
        A base node type for derivation of: Network, Station and Channel
        types.

    The parent class for the network, station and channel classes.
    """
    def __init__(self, code, description=None, comments=None, start_date=None,
                 end_date=None, restricted_status=None, alternate_code=None,
                 historical_code=None, data_availability=None):
        """
        :type code: str
        :param code: The SEED network, station, or channel code
        :type description: str, optional
        :param description: A description of the resource
        :type comments: list of :class:`Comment`, optional
        :param comments: An arbitrary number of comments to the resource
        :type start_date: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param start_date: The start date of the resource
        :type end_date: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param end_date: The end date of the resource
        :type restricted_status: str, optional
        :param restricted_status: The restriction status
        :type alternate_code: str, optional
        :param alternate_code: A code used for display or association,
            alternate to the SEED-compliant code.
        :type historical_code: str, optional
        :param historical_code: A previously used code if different from the
            current code.
        :type data_availability: :class:`~obspy.station.util.DataAvailability`
        :param data_availability: Information about time series availability
            for the network/station/channel.
        """
        self.code = code
        self.comments = comments or []
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.restricted_status = restricted_status
        self.alternate_code = alternate_code
        self.historical_code = historical_code
        self.data_availability = data_availability

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, value):
        if not value:
            msg = "A Code is required"
            raise ValueError(msg)
        self._code = str(value).strip()

    @property
    def alternate_code(self):
        """
        From the StationXML definition:
            A code used for display or association, alternate to the
            SEED-compliant code.
        """
        return self._alternate_code

    @alternate_code.setter
    def alternate_code(self, value):
        if value:
            self._alternate_code = value.strip()
        else:
            self._alternate_code = None

    @property
    def historical_code(self):
        """
        From the StationXML definition:
            A previously used code if different from the current code.
        """
        return self._historical_code

    @historical_code.setter
    def historical_code(self, value):
        if value:
            self._historical_code = value.strip()
        else:
            self._historical_code = None

    def copy(self):
        """
        Returns a deepcopy of the object.

        :rtype: same class as original object
        :return: Copy of current object.

        .. rubric:: Examples

        1. Create a station object and copy it

            >>> from obspy import read_inventory
            >>> sta = read_inventory()[0][0]
            >>> sta2 = sta.copy()

           The two objects are not the same:

            >>> sta is sta2
            False

           But they have equal data (before applying further processing):

            >>> sta == sta2
            True

        2. The following example shows how to make an alias but not copy the
           data. Any changes on ``st3`` would also change the contents of
           ``st``.

            >>> sta3 = sta
            >>> sta is sta3
            True
            >>> sta == sta3
            True
        """
        return copy.deepcopy(self)

    def is_active(self, time=None, starttime=None, endtime=None):
        """
        Checks if the item was active at some given point in time (`time`)
        and/or if it was active at some point during a certain time range
        (`starttime`, `endtime`).

        .. note::
            If none of the time constraints is specified the result will always
            be `True`.

        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Only include networks/stations/channels active at given
            point in time.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Only include networks/stations/channels active at or
            after given point in time (i.e. channels ending before given time
            will not be shown).
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only include networks/stations/channels active before
            or at given point in time (i.e. channels starting after given time
            will not be shown).
        :rtype: bool
        :returns: `True`/`False` depending on whether the item matches the
            specified time criteria.
        """
        if time is not None:
            if self.start_date is not None and time < self.start_date:
                return False
            if self.end_date is not None and time > self.end_date:
                return False
        if starttime is not None and self.end_date is not None:
            if starttime > self.end_date:
                return False
        if endtime is not None and self.start_date is not None:
            if endtime < self.start_date:
                return False

        return True


class DataAvailability(ComparingObject):
    """
    A description of time series data availability. This information should
    be considered transient and is primarily useful as a guide for
    generating time series data requests. The information for a
    DataAvailability (time) span may be specific to the time range used in a
    request that resulted in the document or limited to the availability of
    data within the request range. These details may or may not be
    retained when synchronizing metadata between data centers.
    """
    def __init__(self, start, end):
        self.start = UTCDateTime(start)
        self.end = UTCDateTime(end)

    def __str__(self):
        return "Data Availability from %s to %s." % (str(self.start),
                                                     str(self.end))

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Equipment(ComparingObject):
    """
    An object containing a detailed description of an equipment.
    """
    def __init__(self, type=None, description=None, manufacturer=None,
                 vendor=None, model=None, serial_number=None,
                 installation_date=None, removal_date=None,
                 calibration_dates=None, resource_id=None):
        """
        :type type: str
        :param type: The equipment type
        :type description: str
        :param description: Description of the equipment
        :type manufacturer: str
        :param manufacturer: The manufacturer of the equipment
        :type vendor: str
        :param vendor: The vendor of the equipment
        :type model: str
        :param model: The model of the equipment
        :type serial_number: str
        :param serial_number: The serial number of the equipment
        :type installation_date: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param installation_date: The installation date of the equipment
        :type removal_date: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param removal_date: The removal data of the equipment
        :type calibration_dates: list of
            :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param calibration_dates: A list with all calibration dates of the
            equipment.
        :type resource_id: str
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the data center/software that generated
            the document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behavior equipment with the
            same ID should contain the same information/be derived from the
            same base instruments.
        """
        self.type = type
        self.description = description
        self.manufacturer = manufacturer
        self.vendor = vendor
        self.model = model
        self.serial_number = serial_number
        self.installation_date = installation_date
        self.removal_date = removal_date
        self.calibration_dates = calibration_dates or []
        self.resource_id = resource_id

        @property
        def installation_date(self):
            return self._installation_date

        @installation_date.setter
        def installation_date(self, value):
            if value is None or isinstance(value, UTCDateTime):
                self._installation_date = value
                return
            self._installation_date = UTCDateTime(value)

        @property
        def removal_date(self):
            return self._removal_date

        @removal_date.setter
        def removal_date(self, value):
            if value is None or isinstance(value, UTCDateTime):
                self._removal_date = value
                return
            self._removal_date = UTCDateTime(value)

    def __str__(self):
        ret = ("Equipment:\n"
               "\tType: {type}\n"
               "\tDescription: {description}\n"
               "\tManufacturer: {manufacturer}\n"
               "\tVendor: {vendor}\n"
               "\tModel: {model}\n"
               "\tSerial number: {serial_number}\n"
               "\tInstallation date: {installation_date}\n"
               "\tRemoval date: {removal_date}\n"
               "\tResource id: {resource_id}\n"
               "\tCalibration Dates:\n")
        for calib_date in self.calibration_dates:
            ret += "\t\t%s\n" % calib_date
        ret = ret.format(**self.__dict__)
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Operator(ComparingObject):
    """
    An operating agency and associated contact persons. If there are multiple
    operators, each one should be encapsulated within an Operator object. Since
    the Contact element is a generic type that represents any contact person,
    it also has its own optional Agency element.
    """
    def __init__(self, agencies, contacts=None, website=None):
        """
        :type agencies: list of str
        :param agencies: The agencies of the operator.
        :type contacts: list of :class:`Person`, optional
        :param contacts: One or more contact persons.
        :type website: str, optional
        :param website: The website.
        """
        self.agencies = agencies
        self.contacts = contacts or []
        self.website = website

    @property
    def agencies(self):
        return self._agencies

    @agencies.setter
    def agencies(self, value):
        if not hasattr(value, "__iter__") or len(value) < 1:
            msg = ("agencies needs to be iterable, e.g. a list, and contain "
                   "at least one entry.")
            raise ValueError(msg)
        self._agencies = value

    @property
    def contacts(self):
        return self._contacts

    @contacts.setter
    def contacts(self, value):
        if not hasattr(value, "__iter__"):
            msg = ("contacts needs to be iterable, e.g. a list.")
            raise ValueError(msg)
        self._contacts = value


class Person(ComparingObject):
    """
    From the StationXML definition:
        Representation of a person's contact information. A person can belong
        to multiple agencies and have multiple email addresses and phone
        numbers.
    """
    email_pattern = re.compile("[\w\.\-_]+@[\w\.\-_]+")

    def __init__(self, names=None, agencies=None, emails=None, phones=None):
        """
        :type names: list of str, optional
        :param names: Self-explanatory. Multiple names allowed.
        :type agencies: list of str, optional
        :param agencies: Self-explanatory. Multiple agencies allowed.
        :type emails: list of str, optional
        :param emails: Self-explanatory. Multiple emails allowed.
        :type phones: list of :class:`PhoneNumber`, optional
        :param phones: Self-explanatory. Multiple phone numbers allowed.
        """
        self.names = names or []
        self.agencies = agencies or []
        self.emails = emails or []
        self.phones = phones or []

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        if not hasattr(value, "__iter__"):
            msg = "names needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self._names = value

    @property
    def agencies(self):
        return self._agencies

    @agencies.setter
    def agencies(self, value):
        if not hasattr(value, "__iter__"):
            msg = "agencies needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self._agencies = value

    @property
    def emails(self):
        return self._emails

    @emails.setter
    def emails(self, values):
        if not hasattr(values, "__iter__"):
            msg = "emails needs to be iterable, e.g. a list."
            raise ValueError(msg)
        for value in values:
            if re.match(self.email_pattern, value) is None:
                msg = ("emails needs to match the pattern "
                       "'[\w\.\-_]+@[\w\.\-_]+'")
                raise ValueError(msg)
        self._emails = values

    @property
    def phones(self):
        return self._phones

    @phones.setter
    def phones(self, values):
        if not hasattr(values, "__iter__"):
            msg = "phones needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self._phones = values


class PhoneNumber(ComparingObject):
    """
    A simple object representing a phone number.
    """
    phone_pattern = re.compile("^[0-9]+-[0-9]+$")

    def __init__(self, area_code, phone_number, country_code=None,
                 description=None):
        """
        :type area_code: int
        :param area_code: The area code.
        :type phone_number: str
        :param phone_number: The phone number minus the country and area code.
            Must be in the form "[0-9]+-[0-9]+", e.g. 1234-5678.
        :type country_code: int, optional
        :param country_code: The country code.
        :type description: str, optional
        :param description: Any additional information.
        """
        self.country_code = country_code
        self.area_code = area_code
        self.phone_number = phone_number
        self.description = description

    @property
    def phone_number(self):
        return self._phone_number

    @phone_number.setter
    def phone_number(self, value):
        if re.match(self.phone_pattern, value) is None:
            msg = "phone_number needs to match the pattern '[0-9]+-[0-9]+'"
            raise ValueError(msg)
        self._phone_number = value


class ExternalReference(ComparingObject):
    """
    From the StationXML definition:
        This type contains a URI and description for external data that users
        may want to reference in StationXML.
    """
    def __init__(self, uri, description):
        """
        :type uri: str
        :param uri: The URI to the external data.
        :type description: str
        :param description: A description of the external data.
        """
        self.uri = uri
        self.description = description


class Comment(ComparingObject):
    """
    From the StationXML definition:
        Container for a comment or log entry. Corresponds to SEED blockettes
        31, 51 and 59.
    """
    def __init__(self, value, id=None, begin_effective_time=None,
                 end_effective_time=None, authors=None):
        """
        :type value: str
        :param value: The actual comment string
        :type id: int
        :param id: ID of comment, must be 0 or greater.
        :type begin_effective_time:
            :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param begin_effective_time: The effective start date.
        :type end_effective_time:
            :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param end_effective_time: The effective end date.
        :type authors: list of :class:`Person`, optional
        :param authors: The authors of this comment.
        """
        self.value = value
        self.begin_effective_time = begin_effective_time
        self.end_effective_time = end_effective_time
        self.authors = authors or []
        self.id = id

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        if value is None:
            self._id = value
            return
        if not int(value) >= 0:
            msg = "ID must be 0 or positive integer."
            raise ValueError(msg)
        self._id = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = str(value)

    @property
    def begin_effective_time(self):
        return self._begin_effective_time

    @begin_effective_time.setter
    def begin_effective_time(self, value):
        if value is None:
            self._begin_effective_time = None
            return
        self._begin_effective_time = UTCDateTime(value)

    @property
    def end_effective_time(self):
        return self._end_effective_time

    @end_effective_time.setter
    def end_effective_time(self, value):
        if value is None:
            self._end_effective_time = None
            return
        self._end_effective_time = UTCDateTime(value)

    @property
    def authors(self):
        return self._authors

    @authors.setter
    def authors(self, values):
        if not hasattr(values, "__iter__"):
            msg = "authors needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self._authors = values


class Site(ComparingObject):
    """
    From the StationXML definition:
        Description of a site location using name and optional geopolitical
        boundaries (country, city, etc.).
    """
    def __init__(self, name, description=None, town=None, county=None,
                 region=None, country=None):
        """
        :type name: str
        :param name: The commonly used name of this station, equivalent to the
            SEED blockette 50, field 9.
        :type description: str, optional
        :param description: A longer description of the location of this
            station, e.g.  "NW corner of Yellowstone National Park" or "20
            miles west of Highway 40."
        :type town: str, optional
        :param town: The town or city closest to the station.
        :type county: str, optional
        :param county: The county.
        :type region: str, optional
        :param region: The state, province, or region of this site.
        :type country: str, optional
        :param country: The country.
        """
        self.name = name
        self.description = description
        self.town = town
        self.county = county
        self.region = region
        self.country = country

    def __str__(self):
        ret = ("Site: {name}\n"
               "\tDescription: {description}\n"
               "\tTown:    {town}\n"
               "\tCounty:  {county}\n"
               "\tRegion:  {region}\n"
               "\tCountry: {country}")
        ret = ret.format(
            name=self.name, description=self.description,
            town=self.town, county=self.county, region=self.region,
            country=self.country)
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Latitude(FloatWithUncertaintiesFixedUnit):
    """
    Latitude object

    :type value: float
    :param value: Latitude value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    :type datum: str
    :param datum: Datum for latitude coordinate
    """
    _minimum = -90
    _maximum = 90
    _unit = "DEGREES"

    def __init__(self, value, lower_uncertainty=None, upper_uncertainty=None,
                 datum=None):
        """
        """
        self.datum = datum
        super(Latitude, self).__init__(
            value, lower_uncertainty=lower_uncertainty,
            upper_uncertainty=upper_uncertainty)


class Longitude(FloatWithUncertaintiesFixedUnit):
    """
    Longitude object

    :type value: float
    :param value: Longitude value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    :type datum: str
    :param datum: Datum for longitude coordinate
    """
    _minimum = -180
    _maximum = 180
    unit = "DEGREES"

    def __init__(self, value, lower_uncertainty=None, upper_uncertainty=None,
                 datum=None):
        """
        """
        self.datum = datum
        super(Longitude, self).__init__(
            value, lower_uncertainty=lower_uncertainty,
            upper_uncertainty=upper_uncertainty)


class Distance(FloatWithUncertaintiesAndUnit):
    """
    Distance object

    :type value: float
    :param value: Distance value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    :type unit: str
    :param unit: Unit for distance measure.
    """
    def __init__(self, value, lower_uncertainty=None, upper_uncertainty=None,
                 unit="METERS"):
        super(Distance, self).__init__(
            value, lower_uncertainty=lower_uncertainty,
            upper_uncertainty=upper_uncertainty)
        self._unit = unit


class Azimuth(FloatWithUncertaintiesFixedUnit):
    """
    Azimuth object

    :type value: float
    :param value: Azimuth value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    """
    _minimum = 0
    _maximum = 360
    unit = "DEGREES"


class Dip(FloatWithUncertaintiesFixedUnit):
    """
    Dip object

    :type value: float
    :param value: Dip value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    """
    _minimum = -90
    _maximum = 90
    unit = "DEGREES"


class ClockDrift(FloatWithUncertaintiesFixedUnit):
    """
    ClockDrift object

    :type value: float
    :param value: ClockDrift value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    """
    _minimum = 0
    unit = "SECONDS/SAMPLE"


class SampleRate(FloatWithUncertaintiesFixedUnit):
    """
    SampleRate object

    :type value: float
    :param value: ClockDrift value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    """
    unit = "SAMPLES/S"


class Frequency(FloatWithUncertaintiesFixedUnit):
    """
    Frequency object

    :type value: float
    :param value: Frequency value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    """
    unit = "HERTZ"


class Angle(FloatWithUncertaintiesFixedUnit):
    """
    Angle object

    :type value: float
    :param value: Angle value
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    """
    _minimum = -360
    _maximum = 360
    unit = "DEGREES"


def _unified_content_strings(contents):
    contents_unique = sorted(set(contents), key=_seed_id_keyfunction)
    contents_counts = [
        (item, contents.count(item)) for item in contents_unique]
    items = [item if count == 1 else "{} ({}x)".format(item, count)
             for item, count in contents_counts]
    return items


# make TextWrapper only split on colons, so that we avoid splitting in between
# e.g. network code and network code occurence count (can be controlled with
# class attributes).
# Also avoid lines starting with ", " (need to patch the class for this)
class InventoryTextWrapper(TextWrapper):
    wordsep_re = re.compile(r'(, )')
    wordsep_simple_re = re.compile(r'(, )')

    def _wrap_chunks(self, *args, **kwargs):
        # the following doesn't work somehow (likely because of future??)
        # lines = super(InventoryTextWrapper, self)._wrap_chunks(
        #     *args, **kwargs)
        lines = TextWrapper._wrap_chunks(self, *args, **kwargs)
        lines = [re.sub(r'([\b\s]+), (.*)', r'\1\2', line, count=1)
                 for line in lines]
        return lines


def _textwrap(text, *args, **kwargs):
    return InventoryTextWrapper(*args, **kwargs).wrap(text)


def _seed_id_keyfunction(x):
    """
    Keyfunction to use in sorting two (partial) SEED IDs

    Assumes that the last (or only) "."-separated part is a channel code.
    Assumes the last character is a the component code and sorts it
    "Z"-"N"-"E"-others_lexical.
    """
    # for comparison we build a list of 5 SEED code pieces:
    # [network, station, location, band+instrument, component]
    # with partial codes (i.e. not 4 fields after splitting at dots),
    # we go with the following assumptions (these seem a bit random, but that's
    # what can be encountered in string representations of the Inventory object
    # hierarchy):
    #  - no dot means network code only (e.g. "IU")
    #  - one dot means network.station code only (e.g. "IU.ANMO")
    #  - two dots means station.location.channel code only (e.g. "ANMO.10.BHZ")
    #  - three dots: full SEED ID (e.g. "IU.ANMO.10.BHZ")
    #  - more dots: sort after any of the previous, plain lexical sort
    # if no "." in the string: assume it's a network code

    # split to get rid of the description that that is added to networks and
    # stations which might also contain dots.
    number_of_dots = x.strip().split()[0].count(".")

    x = x.upper()
    if number_of_dots == 0:
        x = [x] + [""] * 4
    elif number_of_dots == 1:
        x = x.split(".") + [""] * 3
    elif number_of_dots in (2, 3):
        x = x.split(".")
        if number_of_dots == 2:
            x = [""] + x
        # split channel code into band+instrument code and component code
        x = x[:-1] + [x[-1][:-1], x[-1] and x[-1][-1] or '']
        # special comparison for component code, convert "ZNE" to integers
        # which compare less than any character
        comp = "ZNE".find(x[-1])
        # last item is component code, either the original 1-char string, or an
        # int from 0-2 if any of "ZNE". Python3 does not allow comparison of
        # int and string anymore (Python 2 always compares ints smaller than
        # any string), so we need to work around this by making this last item
        # a tuple with first item False for ints and True for strings.
        if comp >= 0:
            x[-1] = (False, comp)
        else:
            x[-1] = (True, x[-1])
    # all other cases, just convert the upper case string to a single item
    # list - it will compare greater than any of the split lists.
    else:
        x = [x, ]

    return x


def plot_inventory_epochs(plot_dict, outfile=None, colorspace=None, show=True,
                          combine=True):
    """
    Creates a plot from inventory object's epoch plottable structure.
    :param plot_dict: Dictionary of inventory epochs. A structure used and
        created by the given inventory/station/network/channel object that
        contains the epoch data for an inventory in a tuple with the format
        (start_time, end_time, sub_dict) where sub_dict is a dictionary with
        the same type of structure for any subcomponents.
    :type plot_dict: dict
    :param outfile: If this parameter is included, the plot will be saved to
        a file with the given filename.
    :type outfile: str
    :param colorspace: If this parameter is included, the plot will use the
        given colorspace for inventory plotting
    :type colorspace: matplotlib.colors.LinearSegmentedColormap
    :param show: If set as true, will display the plot in a window
    :type show: boolean
    :param combine: If set as true, channels with matching epochs will be set
        to use the same y-axis values
    :type combine: boolean
    """

    import matplotlib.pyplot as plt

    # y dictionary will hold plot component's initial y-value & height
    y_dict = {}
    # used to combine y-axis values for data with matching epochs
    mg_dict = {}
    if combine:
        mg_dict = _combine_same_epochs(plot_dict)
    # need to do tree traversal for getting appropriate y-axis variables
    _plot_traversal_helper(plot_dict, y_dict, mg_dict)

    # get height of each inv. component data, color lines according to height
    y_tick_labels = []
    y_ticks = []
    y_min = float('inf')
    y_max = 0
    max_lbl_len = 0
    clr_dict = {}
    clr_grps = []
    for key in sorted(y_dict.keys()):
        (tick, height) = y_dict[key]
        if height == 1:
            y_tick_labels.append(key)
            y_ticks.append(tick)
            y_min = min(tick-1, y_min)
            y_max = max(tick+height, y_max)
            max_lbl_len = max(max_lbl_len, len(key))
        elif '.' in key:
            # only color groupings one level above base (i.e., channel) level
            clr_grps.append(key)

    # initialize plot parameters
    fig = plt.figure(figsize=(5+(.1*max_lbl_len), .25 * y_max))
    ax = plt.gca()
    # set yticks according to plot dictionary, xaxis according to date objects
    plt.yticks(y_ticks, y_tick_labels)
    _set_xaxis_obspy_dates(ax)

    if colorspace is None:
        colorspace = cm.Dark2
    clrs = iter(colorspace(linspace(0, 1, len(clr_grps))))
    for grp in clr_grps:
        c = next(clrs)
        clr_dict[grp] = c
    for label in ax.get_yticklabels():
        f = label.get_fontproperties()
        f.set_family('monospace')
    now = UTCDateTime.now().matplotlib_date
    # get the plot ranges
    xmax = 0
    xmin = now
    # add the plottable data to the plot
    (xmin, xmax) = _plot_builder(fig, ax, plot_dict, y_dict, xmin, xmax,
                                 clr_dict, mg_dict)
    xmax = min(xmax, now)
    ax.set_title("Inventory Epoch Plot")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_min, y_max)

    # plt.grid()

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    return fig


def _combine_same_epochs(plot_dict):
    # if multiple inventory objects at station level have matching epochs,
    # merge them into a single line to be plotted and produce new label that
    # identifies all possible subcomponents based on that
    merge_dict, epochs_dict = _merge_epochs(plot_dict)
    return _create_same_epochs_string(merge_dict, epochs_dict)


def _merge_epochs(plot_dict, prefix=''):
    # merges stations that have the same epochs into a single plot axis
    merge_dict = {}
    epochs_dict = {}
    for key in plot_dict.keys():
        label = ''
        if len(prefix) > 0:
            label = prefix + '.'
        label += str(key)
        # only combine at the station level
        (epochs, samp, sub_dict) = plot_dict[key]
        if len(sub_dict) > 0:
            partial_merge, partial_epochs = _merge_epochs(sub_dict,
                                                          prefix=label)
            # multiple stations? different names
            merge_dict.update(partial_merge)
            for key in partial_epochs.keys():
                epochs_dict[key] = partial_epochs[key]
        else:
            # append the epoch of current objects
            ep_tup = []
            for (start, end) in sorted(epochs):
                ep_tup + [start.datetime, end.datetime]
            ep_tup = tuple(ep_tup)
            key = (prefix, ep_tup, samp)
            if key in epochs_dict.keys():
                epochs_dict[key].append(label)
            else:
                epochs_dict[key] = [label]
    return merge_dict, epochs_dict


def _create_same_epochs_string(merge_dict, epochs_dict):
    # used to collect data with matching epochs to abbreviate large inv. plots
    for key in epochs_dict.keys():
        merged = sorted(epochs_dict[key])
        match = merged[0]
        if len(merged) > 1:
            sep = '.'
            # set of data for different locations ('00','10','',etc.)
            loc_chars = set([])
            # specific components for the characters in a channel
            chars = [set([]), set([]), set([])]
            for name in merged:
                split = name.split(sep)
                loc = split[2]
                # location may be blank, use '_' to identify in merged list
                if loc == '':
                    loc_chars.add('_')
                else:
                    loc_chars.add(loc)
                last = split[3]
                band = last[0]
                inst = last[1]
                ornt = last[2:]
                chars[0].add(band)
                chars[1].add(inst)
                chars[2].add(ornt)
            match = sep.join(name.split(sep)[:-2]) + '.'
            loc_chars = sorted(list(loc_chars))
            if len(loc_chars) == 1:
                match += loc_chars[0]
            else:
                match += '['
                for i in range(len(loc_chars)-1):
                    match += loc_chars[i]
                    match += '|'
                match += loc_chars[len(loc_chars)-1]
                match += ']'
            match += '.'
            # read in each character from the channel character set
            for char_set in chars:
                group = ''
                if len(char_set) > 1:
                    group = '['
                for char in sorted(char_set):
                    group += char
                if len(char_set) > 1:
                    group += ']'
                match += group
            for name in merged:
                merge_dict[name] = match
    return merge_dict


def _merge_plottable_structs(eps1, eps2):
    # each epoch for a given channel, station, etc. is a distinct inventory
    # object; this method merges all inventory epochs for a given name into
    # a single list to (hopefully) make the plotting process more painless
    merged_dict = eps1
    for key in eps2.keys():
        if key not in merged_dict.keys():
            merged_dict[key] = eps2[key]
        else:
            (epochs_1, samp_rate_1, sub_dict_1) = eps1[key]
            (epochs_2, samp_rate_2, sub_dict_2) = eps2[key]

            # sample rate isn't TOO important, though it may vary by epoch
            # we just use the first as a matter of convention for dashing lines
            # (otherwise we wouldn't need to keep track of it)
            epochs = epochs_1 + epochs_2
            sub_dict = _merge_plottable_structs(sub_dict_1, sub_dict_2)
            merged_dict[key] = (epochs, samp_rate_1, sub_dict)

            """if samp_rate_1 == samp_rate_2:
            else:
                key_name_1 = key + "-" + samp_rate_1
                key_name_2 = key + "-" + samp_rate_2
                merged_dict[key_name_1] = merged_dict.pop(key)
                merged_dict[key_name_2] = (epochs_2, samp_rate_2, sub_dict_2)
            """
    return merged_dict


def _plot_traversal_helper(plot_dict, y_dict, mg_dict, offset=0, prefix=''):
    # recursively get proper spacing for given structure
    # using the sub-dictionaries for each
    # sorting allows networks and stations to have their data be grouped
    sorted_keys = sorted(plot_dict.keys())
    for key in sorted_keys:
        # get the number of total sub-components
        # add 1 to get the height of the bounding
        # then add 1 again to get the y_tick for the next key
        label = ''
        if len(prefix) > 0:
            label = prefix + '.'
        label += key
        # assign the current data to an axis value if it isn't already
        # (necessary because epoch boundaries can contain same data)
        # and then prevent collisions on y-axis values
        current_offset = offset  # y-axis value to put the current key
        height = 0
        if label in mg_dict.keys():
            y_label = mg_dict[label]
        else:
            y_label = label
        # if current label has already been established, ignore
        # since we've already merged separate epochs of inventory
        if y_label not in y_dict.keys():
            offset += 1
            (_, _, sub_dict) = plot_dict[key]
            offset = _plot_traversal_helper(sub_dict, y_dict, mg_dict,
                                            offset=offset, prefix=y_label)
            if height == 0:
                height = offset - current_offset
            y_dict[y_label] = (current_offset, height)
    return offset


def _plot_builder(fig, ax, plot_dict, y_dict, xmin, xmax, clrs, mg, pfx=''):

    # private method to add lines and rectangles to a given plot object
    import matplotlib.pyplot as plt

    # offsets to put the line markers' tips at ends of lines
    ms = 10  # size of the line marker caps (arrows)
    mark_align_left = tf.offset_copy(ax.transData, fig,
                                     ms/2, units='points')
    mark_align_right = tf.offset_copy(ax.transData, fig,
                                      -ms/2, units='points')
    # sorted_keys = sorted(plot_dict.keys())
    plotted_labels = set([])
    for key in plot_dict.keys():
        # get the number of total sub-components
        # add 1 to get the height of the bounding
        # then add 1 again to get the y_tick for the next key
        label = ''
        if len(pfx) > 0:
            label = pfx + '.'
        label += key
        if label in mg.keys():
            label = mg[label]
        if label in plotted_labels:
            # skip over the duplicated channels
            continue
        (epoch_list, samp_rate, sub_dict) = plot_dict[key]
        for (start_date, end_date) in epoch_list:
            start = start_date.matplotlib_date
            end = end_date.matplotlib_date
            if start != end:
                xmin = min(xmin, start)
            xmax = max(xmax, end)
            (y, height) = y_dict[label]
            # get range of subcomponents
            (temp_xmin, temp_xmax) = _plot_builder(fig, ax, sub_dict, y_dict,
                                                   xmin, xmax, clrs, mg,
                                                   pfx=label)
            if height == 1:
                line_len = 1
                if samp_rate < 100 and samp_rate > 0:
                    line_len *= (100 / samp_rate)
                dash = [line_len, 2]
                # c = clrs[label]
                l, = ax.plot([start, end], [y, y], '--', lw=3, color='k')
                l.set_dashes(dash)
                # left-facing arrow at the start of the epoch
                ax.plot(start, y, marker='>', markersize=ms, color='k',
                        linestyle='none', transform=mark_align_left)
                # right-facing arrow at the end of the epoch
                ax.plot(end, y, marker='<', markersize=ms, color='k',
                        linestyle='none', transform=mark_align_right)
                # plt.gca().add_line(line)
            elif label in clrs.keys() and not (start_date == end_date):
                c = clrs[label]
                # if network epoch not defined, don't bother drawing it
                rect = plt.Rectangle((start, y), end-start, height, fill=True,
                                     lw=2, color=c, alpha=0.2, label=label)
                ax.add_patch(rect)
            xmin = min(xmin, temp_xmin)
            xmax = max(xmax, temp_xmax)
            plotted_labels.add(label)
    return (xmin, xmax)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
