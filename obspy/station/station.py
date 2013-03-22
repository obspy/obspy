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
from obspy.station.util import BaseNode, Equipment


class SeismicStation(BaseNode):
    """
    From the StationXML definition:
        This type represents a Station epoch. It is common to only have a
        single station epoch with the station's creation and termination dates
        as the epoch start and end dates.
    """
    def __init__(self, *args, **kwargs):
        """
        :param latitude:
        :param longitude:
        :param elevation:
        :param site: These fields describe the location of the station using
            geopolitical entities (country, city, etc.).
        :param vault: Type of vault, e.g. WWSSN, tunnel, transportable array,
            etc
        :param geology: Type of rock and/or geologic formation.
        :param equiment: Equipment used by all channels at a station.
        :type operator: A list of strings
        :param operator: An operating agency and associated contact persons. If
            there multiple operators, each one should be encapsulated within an
            Operator tag. Since the Contact element is a generic type that
            represents any contact person, it also has its own optional Agency
            element.
        """
        self.equipment = kwargs.get("equipment", None)
        self.operator = kwargs.get("operator", [])
        super(SeismicStation, self).__init__(*args, **kwargs)

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
