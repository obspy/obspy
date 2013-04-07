#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the SeismicStation class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime
from obspy.station import BaseNode, Equipment, ExternalReference


class SeismicStation(BaseNode):
    """
    From the StationXML definition:
        This type represents a Station epoch. It is common to only have a
        single station epoch with the station's creation and termination dates
        as the epoch start and end dates.
    """
    def __init__(self, code, latitude, longitude, elevation, channels=[],
            **kwargs):
        """
        :type channels: A list of 'obspy.station.SeismicChannel`
        :param channels: All channels belonging to this station.
        :param latitude: The latitude of the station
        :param longitude: The longitude of the station
        :param elevation: The elevation of the station in meter.
        :param site: These fields describe the location of the station using
            geopolitical entities (country, city, etc.).
        :param vault: Type of vault, e.g. WWSSN, tunnel, transportable array,
            etc
        :param geology: Type of rock and/or geologic formation.
        :param equiment: Equipment used by all channels at a station.
        :type operators: A list of `obspy.stations.Operators`
        :param operator: An operating agency and associated contact persons. If
            there multiple operators, each one should be encapsulated within an
            Operator tag. Since the Contact element is a generic type that
            represents any contact person, it also has its own optional Agency
            element.
        :type creation_date: `obspy.UTCDateTime`
        :param creation_date: Date and time (UTC) when the station was first
            installed
        :type termination_date: `obspy.UTCDateTime`
        :param termination_date: Date and time (UTC) when the station was
            terminated or will be terminated. A blank value should be assumed
            to mean that the station is still active. Optional
        :type total_number_of_channels: Integer
        :param total_number_of_channels: Total number of channels recorded at
            this station. Optional.
        :type selected_number_of_channels: Integer
        :param selected_number_of_channels: Number of channels recorded at this
            station and selected by the query that produced this document.
            Optional.
        :type external_references: list of `obspy.station.ExternalReference`
        :param external_references: URI of any type of external report, such as
            IRIS data reports or dataless SEED volumes. Optional.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.channels = channels
        self.site = kwargs.get("site", None)
        self.vault = kwargs.get("vault", None)
        self.geology = kwargs.get("geology", None)
        self.equipments = kwargs.get("equipment", [])
        self.operators = kwargs.get("operators", [])
        self.creation_date = kwargs.get("creation_date", None)
        self.termination_date = kwargs.get("termination_date", None)
        self.total_number_of_channels = kwargs.get("total_number_of_channels",
            None)
        self.selected_number_of_channels = \
            kwargs.get("selected_number_of_channels", None)
        self.external_references = kwargs.get("external_references", [])
        super(SeismicStation, self).__init__(code, **kwargs)

    @property
    def operator(self):
        return self.__operator

    @operator.setter
    def operator(self, value):
        if not hasattr(value, "__iter__"):
            msg = "Operator needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__operator = value

    @property
    def equipment(self):
        return self.__equipment

    @equipment.setter
    def equipment(self, value):
        if value is None or isinstance(value, Equipment):
            self.__equipment = value
        elif isinstance(value, dict):
            self.__equipment = Equipment(**value)
        else:
            msg = ("equipment needs to be be of type obspy.station.Equipment "
                "or contain a dictionary with values suitable for "
                "initialization.")
            raise ValueError

    @property
    def creation_date(self):
        return self.__creation_date

    @creation_date.setter
    def creation_date(self, value):
        if value is None:
            self.__creation_date = None
            return
        if not isinstance(value, UTCDateTime):
            value = UTCDateTime(value)
        self.__creation_date = value

    @property
    def termination_date(self):
        return self.__termination_date

    @termination_date.setter
    def termination_date(self, value):
        if value is not None and not isinstance(value, UTCDateTime):
            value = UTCDateTime(value)
        self.__termination_date = value

    @property
    def external_references(self):
        return self.__external_references

    @external_references.setter
    def external_references(self, value):
        if not hasattr(value, "__iter__"):
            msg = "external_references needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__external_references = value
