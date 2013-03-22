#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functionality.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime


class BaseNode(object):
    """
    From the StationXML definition:
        A base node type for derivation from: Network, Station and Channel
        types.Station
    """
    def __init__(self, code):
        self.comments = kwargs.get("comments", [])

        self.description = kwargs.get("description", None)
        self.code = kwargs.get("code")
        self.starttime = kwargs.get("starttime", None)
        self.endtime = kwargs.get("endtime", None)
        self.restricted_status = kwargs.get("restricted_status", None)
        self.alternate_code = kwargs.get("alternate_code", None)
        self.historical_code = kwargs.get("historical_code", None)

    @property
    def code(self):
        return self.__code

    @code.setter
    def code(self, value):
        if not value:
            msg = "A Code is required"
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


class Equipment(object):
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
        :type calibration_date: list of `obspy.UTCDateTime`
        :param calibration_date: A list with all calibration dates of the
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
        self.removal_date = kwargs.get("removal_data", None)
        self.calibration_date = kwargs.get("calibration_date", [])
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
        def installation_date(self, value):
            if value is None or isinstance(value, UTCDateTime):
                self.__removal_date = value
                return
            self.__removal_date = UTCDateTime(value)
