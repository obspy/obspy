# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from obspy.xseed import DEFAULT_XSEED_VERSION, utils
from obspy.xseed.fields import Integer, Loop

import io
from lxml.etree import Element
import os
import warnings


class BlocketteLengthException(Exception):
    """
    Wrong blockette length detected.
    """
    pass


class BlocketteParserException(Exception):
    """
    General Blockette Parser Exception.
    """
    pass


class Blockette(object):
    """
    General blockette handling.
    """
    # default field for each blockette
    fields = []
    default_fields = [
        Integer(1, "Blockette type", 3),
        Integer(2, "Length of blockette", 4, optional=True)
    ]

    def __init__(self, **kwargs):
        self.debug = kwargs.get('debug', False)
        self.strict = kwargs.get('strict', False)
        self.compact = kwargs.get('compact', False)
        self.record_type = kwargs.get('record_type', None)
        self.record_id = kwargs.get('record_id', None)
        self.blockette_id = "%03d" % self.id
        self.blockette_name = utils.toTag(self.name)
        # debug
        if self.debug:
            print("----")
            print(str(self))
        # filter versions specific fields
        self.xseed_version = kwargs.get('xseed_version', DEFAULT_XSEED_VERSION)
        self.seed_version = kwargs.get('version', 2.4)

    def __str__(self):
        """
        Pretty prints the informations stored in the blockette.
        """
        temp = 'Blockette %s: %s Blockette' % (
            self.blockette_id, utils.toString(self.blockette_name)) + \
            os.linesep
        keys = self.__dict__.keys()
        keys = sorted(keys)
        for key in keys:
            if key in utils.IGNORE_ATTR:
                continue
            temp += '%30s: %s' % (utils.toString(key), self.__dict__[key])
            temp += os.linesep
        return temp.strip()

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def getFields(self, xseed_version=DEFAULT_XSEED_VERSION):
        fields = []
        for field in self.fields:
            # Check XML-SEED version
            if field.xseed_version and \
               field.xseed_version != xseed_version:
                continue
            # Check SEED version
            if field.seed_version and field.seed_version > self.seed_version:
                continue
            fields.append(field)
        return fields

    def parseSEED(self, data, expected_length=0):
        """
        Parse given data for blockette fields and create attributes.
        """
        # convert to stream for test issues
        if isinstance(data, bytes):
            expected_length = len(data)
            data = io.BytesIO(data)
        elif isinstance(data, (str, native_str)):
            raise TypeError("data must be bytes, not string")
        start_pos = data.tell()
        # debug
        if self.debug:
            print(' DATA: %s' % (data.read(expected_length)))
            data.seek(-expected_length, 1)
        blockette_fields = self.default_fields + self.getFields()
        # loop over all blockette fields
        for field in blockette_fields:
            # if blockette length reached break with warning
            if data.tell() - start_pos >= expected_length:
                if not self.strict:
                    break
                if isinstance(field, Loop):
                    break
                msg = "End of blockette " + self.blockette_id + " reached " + \
                      "without parsing all expected fields, here: " + \
                      str(field)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    warnings.warn(msg, category=Warning)
                break
            field.parseSEED(self, data)
            if field.id == 2:
                expected_length = field.data
        # strict tests
        if not self.strict:
            return
        # check length
        end_pos = data.tell()
        blockette_length = end_pos - start_pos
        if expected_length == blockette_length:
            return
        # wrong length
        msg = 'Wrong size of Blockette %s (%d of %d) in sequence %06d'
        msg = msg % (self.blockette_id, blockette_length,
                     expected_length, self.record_id or 0)
        if self.strict:
            raise BlocketteLengthException(msg)
        else:
            warnings.warn(msg, category=Warning)

    def getSEED(self):
        """
        Converts the blockette to a valid SEED string and returns it.
        """
        # loop over all blockette fields
        data = b''
        for field in self.getFields():
            data += field.getSEED(self)
        # add blockette id and length
        _head = '%03d%04d' % (self.id, len(data) + 7)
        return _head.encode('ascii', 'strict') + data

    def parseXML(self, xml_doc):
        """
        Reads lxml etree and fills the blockette with the values of it.
        """
        for field in self.getFields(self.xseed_version):
            field.parseXML(self, xml_doc)

    def getXML(self, show_optional=False,
               xseed_version=DEFAULT_XSEED_VERSION):
        """
        Returns a XML document representing this blockette.
        """
        self.xseed_version = xseed_version
        # root element
        xml_doc = Element(self.blockette_name, blockette=self.blockette_id)
        # loop over all blockette fields
        for field in self.getFields(xseed_version=xseed_version):
            node = field.getXML(self)
            xml_doc.extend(node)
        return xml_doc
