#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A class parsing WADL files describing FDSN web services.

There are couple of datacenter specific fixes in here. They are marked by XXX
and should be removed once the datacenters are fully standard compliant.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime
from obspy.fdsn.header import DEFAULT_DATASELECT_PARAMETERS, \
    DEFAULT_STATION_PARAMETERS, DEFAULT_EVENT_PARAMETERS, \
    WADL_PARAMETERS_NOT_TO_BE_PARSED, DEFAULT_TYPES

from lxml import etree
import warnings


class WADLParser(object):
    def __init__(self, wadl_string):
        doc = etree.fromstring(wadl_string)
        self.nsmap = doc.nsmap
        self._ns = self.nsmap.get(None, None)
        self.parameters = {}

        # Get the url.
        url = self._xpath(doc, "/application/resources")[0].get("base")
        if "dataselect" in url:
            self._default_parameters = DEFAULT_DATASELECT_PARAMETERS
            self._wadl_type = "dataselect"
        elif "station" in url:
            self._default_parameters = DEFAULT_STATION_PARAMETERS
            self._wadl_type = "station"
        elif "event" in url:
            self._default_parameters = DEFAULT_EVENT_PARAMETERS
            self._wadl_type = "event"
        else:
            raise NotImplementedError

        # Map short names to long names.
        self._short_to_long_mapping = {}
        for item in self._default_parameters:
            if len(item) == 1:
                continue
            self._short_to_long_mapping[item[1]] = item[0]

        # Retrieve all the parameters.
        parameters = self._xpath(
            doc, "/application/resources/resource/resource/"
            "method[@id='query'][@name='GET']/request/param")
        # XXX: USGS is special right now. They have to make it one layer
        # deeper. Remove once they fix it.
        if not parameters and "usgs" in url.lower():
            parameters = self._xpath(
                doc, "/application/resources/resource/"
                "method[@id='query'][@name='GET']/request/param")
        if not parameters:
            msg = "Could not find any parameters"
            raise ValueError(msg)

        for param in parameters:
            self.add_parameter(param)

        # Raise a warning if some default parameters are not specified.
        missing_params = []
        for param in self._default_parameters:
            if param not in self.parameters.keys():
                missing_params.append(param)
        if missing_params:
            msg = ("The '%s' service at '%s' cannot deal with the following "
                   "required parameters: %s\nThey will not be available "
                   "for any requests. Any attempt to use them will result "
                   "in an error.") % (self._wadl_type, url,
                                      ", ".join(missing_params))
            warnings.warn(msg)

    def add_parameter(self, param_doc):
        name = param_doc.get("name")

        # Map the short to the long names.
        if name in self._short_to_long_mapping:
            name = self._short_to_long_mapping[name]

        # Skip the parameter if should be ignored.
        if name in WADL_PARAMETERS_NOT_TO_BE_PARSED:
            return

        # XXX: Special handling for the USGS event WADL. minlongitude is preset
        # twice... Remove once they fix it.
        if name == "minlongitude" and "minlongitude" in self.parameters:
            name = "maxlongitude"

        style = param_doc.get("style")
        if style != "query":
            msg = "Unknown parameter style '%s' in WADL" % style
            warnings.warn(msg)

        required = self._convert_boolean(param_doc.get("required"))
        if required is None:
            required = False

        param_type = param_doc.get("type")
        if param_type is None:
            # If not given, choose one from the DEFAULT_TYPES dictionary.
            # Otherwise assign a string.
            if name in DEFAULT_TYPES:
                param_type = DEFAULT_TYPES[name]
            else:
                param_type = str
        elif param_type in ["xs:date", "xs:dateTime"]:
            param_type = UTCDateTime
        elif param_type == "xs:string":
            param_type = str
        elif param_type == "xs:double":
            param_type = float
        elif param_type in ["xs:long", "xs:int", "xs:integer"]:
            param_type = int
        elif param_type == "xs:boolean":
            param_type = bool
        else:
            msg = "Unknown parameter type '%s' in WADL." % param_type
            raise ValueError(msg)

        default_value = param_doc.get("default")
        if default_value is not None:
            if param_type == bool:
                default_value = self._convert_boolean(default_value)
            else:
                default_value = param_type(default_value)

        # Parse any possible options.
        options = []
        for option in self._xpath(param_doc, "option"):
            options.append(param_type(option.get("value")))

        doc = ""
        doc_title = ""
        for doc_elem in self._xpath(param_doc, "doc"):
            title = doc_elem.get("title")
            body = doc_elem.text
            doc_title = title
            if body:
                doc = body
            break

        self.parameters[name] = {
            "required": required,
            "type": param_type,
            "options": options,
            "doc_title": doc_title.strip(),
            "doc": doc.strip(),
            "default_value": default_value}

    @staticmethod
    def _convert_boolean(boolean_string):
        """
        Helper function for boolean value conversion.

        >>> WADLParser._convert_boolean("true")
        True
        >>> WADLParser._convert_boolean("True")
        True

        >>> WADLParser._convert_boolean("false")
        False
        >>> WADLParser._convert_boolean("False")
        False

        >>> WADLParser._convert_boolean("something")
        >>> WADLParser._convert_boolean(1)
        """
        try:
            if boolean_string.lower() == "false":
                return False
            elif boolean_string.lower() == "true":
                return True
            else:
                return None
        except:
            return None

    def _xpath(self, doc, expr):
        """
        Simple helper method for using xpaths with the default namespace.
        """
        nsmap = self.nsmap.copy()
        if self._ns is not None:
            default_abbreviation = "default"
            # being paranoid, can happen that another ns goes by that name
            while default_abbreviation in nsmap:
                default_abbreviation = default_abbreviation + "x"
            parts = []
            # insert prefixes for default namespace
            for x in expr.split("/"):
                if x != "" and ":" not in x:
                    x = "%s:%s" % (default_abbreviation, x)
                parts.append(x)
            expr = "/".join(parts)
            # adapt nsmap accordingly
            nsmap.pop(None, None)
            nsmap[default_abbreviation] = self._ns
        return doc.xpath(expr, namespaces=nsmap)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
