#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility objects.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime
from obspy.core.util.base import ComparingObject
import re


class BaseNode(ComparingObject):
    """
    From the StationXML definition:
        A base node type for derivation of: Network, Station and Channel
        types.

    The parent class for the network, station and channel classes.
    """
    def __init__(self, code, description=None, comments=[], start_date=None,
                 end_date=None, restricted_status=None, alternate_code=None,
                 historical_code=None):
        """
        :type code: String
        :param code: The SEED network, station, or channel code
        :type description: String, optional
        :param description: A description of the resource
        :type comments: List of :class:`~obspy.station.util.Comment`, optional
        :param comments: An arbitrary number of comments to the resource
        :type start_date: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param start_date: The start date of the resource
        :type end_date: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param end_date: The end date of the resource
        :type restricted_status: String, optional
        :param restricted_status: The restriction status
        :type alternate_code: String, optional
        :param alternate_code: A code used for display or association,
            alternate to the SEED-compliant code.
        :type historical_code: String, optional
        :param historical_code: A previously used code if different from the
            current code.
        """
        self.code = code
        self.comments = comments
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.restricted_status = restricted_status
        self.alternate_code = alternate_code
        self.historical_code = historical_code

    @property
    def code(self):
        return self.__code

    @code.setter
    def code(self, value):
        if not value:
            msg = "A Code is required"
            raise ValueError(msg)
        self.__code = str(value)

    @property
    def alternate_code(self):
        """
        From the StationXML definition:
            A code used for display or association, alternate to the
            SEED-compliant code.
        """
        return self.__alternate_code

    @alternate_code.setter
    def alternate_code(self, value):
        if value:
            self.__alternate_code = value
        else:
            self.__alternate_code = None

    @property
    def historical_code(self):
        """
        From the StationXML definition:
            A previously used code if different from the current code.
        """
        return self.__historical_code

    @historical_code.setter
    def historical_code(self, value):
        if value:
            self.__historical_code = value
        else:
            self.__historical_code = None


class Equipment(ComparingObject):
    """
    An object containing a detailed description of an equipment.
    """
    def __init__(self, *args, **kwargs):
        """
        :type type: String
        :param type: The equipment type
        :type description: String
        :param description: Description of the equipment
        :type manufacturer: String
        :param manufacturer: The manufactorer of the equipment
        :type vendor: String
        :param vendor: The vendor of the equipment
        :type model: String
        :param model: The model of the equipment
        :type serial_number: String
        :param serial_number: The serial number of the equipment
        :type installation_date: `obspy.UTCDateTime`
        :param installation_date: The installation date of the equipment
        :type removal_date: `obspy.UTCDateTime`
        :param removal_date: The removal data of the equipment
        :type calibration_dates: list of `obspy.UTCDateTime`
        :param calibration_dates: A list with all calibration dates of the
            equipment.
        :type resource_id: String
        :param resource_id: This field contains a string that should serve as a
            unique resource identifier. This identifier can be interpreted
            differently depending on the datacenter/software that generated the
            document. Also, we recommend to use something like
            GENERATOR:Meaningful ID. As a common behaviour equipment with the
            same ID should contains the same information/be derived from the
            same base instruments.
        """
        self.type = kwargs.get("type", None)
        self.description = kwargs.get("description", None)
        self.manufacturer = kwargs.get("manufacturer", None)
        self.vendor = kwargs.get("vendor", None)
        self.model = kwargs.get("model", None)
        self.serial_number = kwargs.get("serial_number", None)
        self.installation_date = kwargs.get("installation_date", None)
        self.removal_date = kwargs.get("removal_date", None)
        self.calibration_dates = kwargs.get("calibration_dates", [])
        self.resource_id = kwargs.get("resource_id", None)

        @property
        def installation_date(self):
            return self.__installation_date

        @installation_date.setter
        def installation_date(self, value):
            if value is None or isinstance(value, UTCDateTime):
                self.__installation_date = value
                return
            self.__installation_date = UTCDateTime(value)

        @property
        def removal_date(self):
            return self.__removal_date

        @removal_date.setter
        def removal_date(self, value):
            if value is None or isinstance(value, UTCDateTime):
                self.__removal_date = value
                return
            self.__removal_date = UTCDateTime(value)


class Operator(ComparingObject):
    """
    An operating agency and associated contact persons. If there multiple
    operators, each one should be encapsulated within an Operator tag. Since
    the Contact element is a generic type that represents any contact person,
    it also has its own optional Agency element.
    """
    def __init__(self, *args, **kwargs):
        """
        :type agency: A list of strings.
        :param agency: The agency of the operator.
        :type contact: A list of `obspy.station.Person`
        :param contact: One or more contact persons, optional
        :type website: String
        :param website: The website, optional
        """
        self.agencies = kwargs.get("agencies")
        self.contacts = kwargs.get("contacts", [])
        self.website = kwargs.get("website", None)

    @property
    def agencies(self):
        return self.__agencies

    @agencies.setter
    def agencies(self, value):
        if not hasattr(value, "__iter__") or len(value) < 1:
            msg = ("agencies needs to iterable, e.g. a list and contain at "
                   "least one entry.")
            raise ValueError(msg)
        self.__agencies = value

    @property
    def contacts(self):
        return self.__contacts

    @contacts.setter
    def contacts(self, value):
        if not hasattr(value, "__iter__"):
            msg = ("contacts needs to iterable, e.g. a list.")
            raise ValueError(msg)
        self.__contacts = value


class Person(ComparingObject):
    """
    From the StationXML definition:
        Representation of a person's contact information. A person can belong
        to multiple agencies and have multiple email addresses and phone
        numbers.
    """
    def __init__(self, **kwargs):
        """
        :type names: list of strings
        :param names: Self-explanatory. Multiple names allowed. Optional.
        :type agencies list of strings
        :param agencies Self-explanatory. Multiple agencies allowed. Optional.
        :type emails: list of strings
        :param emails: Self-explanatory. Multiple emails allowed. Optional.
        :type phones: list of `obspy.stations.PhoneNumber`
        :param phones: Self-explanatory. Multiple phone numbers allowed.
            Optional.
        """
        self.names = kwargs.get("names", [])
        self.agencies = kwargs.get("agencies", [])
        self.emails = kwargs.get("emails", [])
        self.phones = kwargs.get("phones", [])

    @property
    def names(self):
        return self.__names

    @names.setter
    def names(self, value):
        if not hasattr(value, "__iter__"):
            msg = "names needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__names = value

    @property
    def agencies(self):
        return self.__agencies

    @agencies.setter
    def agencies(self, value):
        if not hasattr(value, "__iter__"):
            msg = "agencies needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__agencies = value

    @property
    def emails(self):
        return self.__emails

    @emails.setter
    def emails(self, values):
        if not hasattr(values, "__iter__"):
            msg = "emails needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__emails = values

    @property
    def phones(self):
        return self.__phones

    @phones.setter
    def phones(self, values):
        if not hasattr(values, "__iter__"):
            msg = "phones needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__phones = values


class PhoneNumber(ComparingObject):
    """
    A simple object representing a phone number.
    """
    phone_pattern = re.compile("^[0-9]+-[0-9]+$")

    def __init__(self, **kwargs):
        """
        :type country_code: Integer
        :param country_code: The country code, optional
        :type area_code: Integer
        :param area_code: The area code
        :type phone_number: String in the form "[0-9]+-[0-9]+", e.g. 1234-5678.
        :param phone_number: The phone number minus the country and area code.
        :type description: String
        :param description: Any additional information, optional.
        """
        self.country_code = kwargs.get("country_code", None)
        self.area_code = kwargs.get("area_code")
        self.phone_number = kwargs.get("phone_number")
        self.description = kwargs.get("description", None)

    @property
    def phone_number(self):
        return self.__phone_number

    @phone_number.setter
    def phone_number(self, value):
        if re.match(self.phone_pattern, value) is None:
            msg = "phone_number needs to match the pattern '[0-9]+-[0-9]+'"
            raise ValueError(msg)
        self.__phone_number = value


class ExternalReference(ComparingObject):
    """
    From the StationXML definition:
        This type contains a URI and description for external data that users
        may want to reference in StationXML.
    """
    def __init__(self, uri, description):
        """
        :type uri: String
        :param uri: The URI to the external data.
        :type description: String
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
    def __init__(self, value, begin_effective_time=None,
                 end_effective_time=None, authors=[]):
        """
        :type value: String
        :param value: The actual comment string
        :type begin_effective_time:
            :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param begin_effective_time: The effective start date, Optional.
        :type end_effective_time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param end_effective_time: The effective end date. Optional.
        :type authors: List of :class:`~obspy.station.util.Person` objects.
        :param authors: The authors of this comment. Optional.
        """
        self.value = value
        self.begin_effective_time = begin_effective_time
        self.end_effective_time = end_effective_time
        self.authors = authors

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = str(value)

    @property
    def begin_effective_time(self):
        return self.__begin_effective_time

    @begin_effective_time.setter
    def begin_effective_time(self, value):
        if value is None:
            self.__begin_effective_time = None
            return
        self.__begin_effective_time = UTCDateTime(value)

    @property
    def end_effective_time(self):
        return self.__end_effective_time

    @end_effective_time.setter
    def end_effective_time(self, value):
        if value is None:
            self.__end_effective_time = None
            return
        self.__end_effective_time = UTCDateTime(value)

    @property
    def authors(self):
        return self.__authors

    @authors.setter
    def authors(self, values):
        if not hasattr(values, "__iter__"):
            msg = "authors needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__authors = values


class Site(ComparingObject):
    """
    From the StationXML definition:
        Description of a site location using name and optional geopolitical
        boundaries (country, city, etc.).
    """
    def __init__(self, name, description=None, town=None, county=None,
                 region=None, country=None):
        """
        :type name: String
        :param name: The commonly used name of this station, equivalent to the
            SEED blockette 50, field 9.
        :type description: String, optional
        :param description: A longer description of the location of this
            station, e.g.  "NW corner of Yellowstone National Park" or "20
            miles west of Highway 40."
        :type town: String, optional
        :param town: The town or city closest to the station.
        :type county: String, optional
        :param county: The county.
        :type region: String, optional
        :param region: The state, province, or region of this site.
        :type country: String, optional
        :param country: THe country.
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
