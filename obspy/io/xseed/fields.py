# -*- coding: utf-8 -*-
"""
Helper module containing xseed fields.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import re
import warnings

from lxml.etree import Element, SubElement

from obspy import UTCDateTime
from .utils import (datetime_2_string, SEEDParserException, get_xpath,
                    set_xpath, to_tag)


class SEEDTypeException(Exception):
    pass


class Field(object):
    """
    General SEED field.
    """
    def __init__(self, id, name, *args, **kwargs):  # @UnusedVariable
        # default
        self.id = id
        self.flag = ''
        self.name = name
        self.xseed_version = kwargs.get('xseed_version', None)
        self.seed_version = kwargs.get('version', None)
        self.mask = kwargs.get('mask', None)
        self.xpath = kwargs.get('xpath', None)
        if self.id:
            self.field_id = "F%02d" % self.id
        else:
            self.field_id = None
        self.field_name = kwargs.get('xml_tag', to_tag(self.name))
        self.attribute_name = to_tag(self.name)
        # options
        self.optional = kwargs.get('optional', False)
        self.ignore = kwargs.get('ignore', False)
        self.strict = kwargs.get('strict', False)
        self.compact = kwargs.get('compact', False)
        self.default_value = kwargs.get('default_value', self.default)

    def __str__(self):
        if self.id:
            return "F%02d" % self.id

    def _repr_pretty_(self, p, cycle):  # @UnusedVariable
        p.text(str(self))

    def convert(self, value):
        return value

    def _format_string(self, s):
        """
        Using SEED specific flags to format strings.

        This method is partly adopted from fseed.py, the SEED builder for
        SeisComP written by Andres Heinloo, GFZ Potsdam in 2005.
        """
        if isinstance(s, bytes):
            sn = s.decode('utf-8').strip()
        else:
            sn = str(s).strip()
        if self.flags and 'T' in self.flags:
            if not sn and self.default_value:
                return self.default_value
            return datetime_2_string(sn, self.compact)
        if not self.flags:
            return sn
        rx_list = []
        if 'U' in self.flags:
            # upper case A—Z
            rx_list.append(r"[A-Z]")
        if 'L' in self.flags:
            # lower case a—z
            rx_list.append(r"[a-z]")
        if 'N' in self.flags:
            # digits 0—9
            rx_list.append(r"[0-9]")
        if 'P' in self.flags:
            # any punctuation characters (including “_”)
            rx_list.append(r"[^A-Za-z0-9 ]")
        if 'S' in self.flags:
            # spaces between words
            rx_list.append(r" ")
        if '_' in self.flags:
            # underline symbol
            rx_list.append(r"_")
        # auto-format
        if 'U' in self.flags and 'L' not in self.flags:
            sn = sn.upper()
        elif 'L' in self.flags and 'U' not in self.flags:
            sn = sn.lower()
        rx = "|".join(rx_list)
        sn = "".join(re.findall(rx, sn))
        if re.match("(" + rx + ")*$", sn) is None:
            msg = "Can't convert string %s with flags %s" % (s, self.flags)
            raise SEEDTypeException(msg)
        return sn

    def parse_seed(self, blockette, data):
        """
        """
        try:
            text = self.read(data, strict=blockette.strict)
        except Exception as e:
            if blockette.strict:
                raise e
            # default value if not set
            text = self.default_value
        # check if already exists
        if hasattr(blockette, self.attribute_name):
            temp = getattr(blockette, self.attribute_name)
            if not isinstance(temp, list):
                temp = [temp]
            temp.append(text)
            text = temp
        setattr(blockette, self.attribute_name, text)
        self.data = text
        # debug
        if blockette.debug:
            print('  %s: %s' % (self, text))

    def get_seed(self, blockette, pos=0):
        """
        """
        self.compact = blockette.compact
        try:
            result = getattr(blockette, self.attribute_name)
        except Exception:
            if blockette.strict:
                msg = "Missing attribute %s in Blockette %s"
                raise Exception(msg % (self.name, blockette))
            result = self.default_value
        # watch for multiple entries
        if isinstance(result, list):
            result = result[pos]
        # debug
        if blockette.debug:
            print('  %s: %s' % (self, result))
        return self.write(result, strict=blockette.strict)

    def get_xml(self, blockette, pos=0):
        """
        """
        if self.ignore:
            # debug
            if blockette.debug:
                print('  %s: ignored')
            return []
        try:
            result = getattr(blockette, self.attribute_name)
        except Exception:
            if blockette.strict:
                msg = "Missing attribute %s in Blockette %s"
                raise Exception(msg % (self.name, blockette))
            result = self.default
        # watch for multiple entries
        if isinstance(result, list):
            result = result[pos]
        # optional if empty
        if self.optional:
            try:
                result = result.strip()
            except Exception:
                pass
            if not result:
                # debug
                if blockette.debug:
                    print('  %s: skipped because optional')
                return []
        # reformat float
        if isinstance(self, Float):
            result = self.write(result)
        # Converts to XPath if necessary.
        if self.xpath:
            result = set_xpath(self.xpath, result)
        # create XML element
        node = Element(self.field_name)
        if isinstance(result, bytes):
            node.text = result.decode().strip()
        else:
            node.text = str(result).strip()
        # debug
        if blockette.debug:
            print('  %s: %s' % (self, [node]))
        return [node]

    def parse_xml(self, blockette, xml_doc, pos=0):
        """
        """
        try:
            text = xml_doc.xpath(self.field_name + "/text()")[pos]
        except Exception:
            setattr(blockette, self.attribute_name, self.default_value)
            # debug
            if blockette.debug:
                print('  %s: set to default value %s' % (self,
                                                         self.default_value))
            return
        # Parse X-Path if necessary. The isdigit test assures legacy support
        # for XSEED without XPaths.
        if self.xpath and not text.isdigit():
            text = get_xpath(text)
        # check if already exists
        if hasattr(blockette, self.attribute_name):
            temp = getattr(blockette, self.attribute_name)
            if not isinstance(temp, list):
                temp = [temp]
            temp.append(text)
            text = temp
        setattr(blockette, self.attribute_name, self.convert(text))
        # debug
        if blockette.debug:
            print('  %s: %s' % (self, text))


class Integer(Field):
    """
    An integer field.
    """
    def __init__(self, id, name, length, **kwargs):
        self.default = 0
        Field.__init__(self, id, name, **kwargs)
        self.length = length

    def convert(self, value):
        try:
            if isinstance(value, list):
                return [int(_i) for _i in value]
            else:
                return int(value)
        except Exception:
            if not self.strict:
                return self.default_value
            msg = "No integer value found for %s." % self.attribute_name
            raise SEEDTypeException(msg)

    def read(self, data, strict=False):  # @UnusedVariable
        temp = data.read(self.length)
        return self.convert(temp)

    def write(self, data, strict=False):  # @UnusedVariable
        format_str = "%%0%dd" % self.length
        try:
            temp = int(data)
        except Exception:
            msg = "No integer value found for %s." % self.attribute_name
            raise SEEDTypeException(msg)
        result = format_str % temp
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.attribute_name)
            raise SEEDTypeException(msg)
        return result.encode()


class Float(Field):
    """
    A float number with a fixed length and output mask.
    """
    def __init__(self, id, name, length, **kwargs):
        self.default = 0.0
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        if not self.mask:
            msg = "Float field %s requires a data mask." % self.attribute_name
            raise SEEDTypeException(msg)

    def convert(self, value):
        try:
            if isinstance(value, list):
                return [float(_i) for _i in value]
            else:
                return float(value)
        except Exception:
            if not self.strict:
                return self.default_value
            msg = "No float value found for %s." % self.attribute_name
            raise SEEDTypeException(msg)

    def read(self, data, strict=False):  # @UnusedVariable
        temp = data.read(self.length)
        return self.convert(temp)

    def write(self, data, strict=False):  # @UnusedVariable
        format_str = "%%0%ds" % self.length
        try:
            temp = float(data)
        except Exception:
            msg = "No float value found for %s." % self.attribute_name
            raise SEEDTypeException(msg)
        # special format for exponential output
        result = format_str % (self.mask % temp)
        if 'E' in self.mask or 'e' in self.mask:
            result = result.upper()
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.attribute_name)
            raise SEEDTypeException(msg)
        return result.encode()


class FixedString(Field):
    """
    A string field with a fixed width.
    """
    def __init__(self, id, name, length, flags='', **kwargs):
        self.default = ' ' * length
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.flags = flags

    def read(self, data, strict=False):  # @UnusedVariable
        return self._format_string(data.read(self.length).strip())

    def write(self, data, strict=False):  # @UnusedVariable
        # Leave fixed length alphanumeric fields left justified (no leading
        # spaces), and pad them with spaces (after the field’s contents).
        format_str = "%%-%ds" % self.length
        result = format_str % self._format_string(data)
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.attribute_name)
            raise SEEDTypeException(msg)
        return result.encode()


class VariableString(Field):
    """
    Variable length ASCII string, ending with a tilde: ~ (ASCII 126).

    Variable length fields cannot have leading or trailing spaces. Character
    counts for variable length fields do not include the tilde terminator.
    """
    def __init__(self, id, name, min_length=0, max_length=None, flags='',
                 **kwargs):
        self.default = ''
        Field.__init__(self, id, name, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.flags = flags

    def convert(self, value):
        # check for datetime
        if 'T' in self.flags:
            return UTCDateTime(value)
        return value

    def read(self, data, strict=False):
        data = self._read(data)
        # check for datetime
        if 'T' in self.flags:
            # default value
            if data:
                # create a full SEED date string
                temp = b"0000,000,00:00:00.0000"
                data += temp[len(data):]
                return UTCDateTime(data.decode())
            if self.default_value:
                return self.default_value
            if self.min_length:
                if strict:
                    raise SEEDParserException
                warnings.warn('Date is required.', UserWarning)
            return ""
        else:
            if self.flags:
                return self._format_string(data)
            else:
                return data

    def _read(self, data):
        buffer = b''
        if self.min_length:
            buffer = data.read(self.min_length)
            if b'~' in buffer:
                return buffer.split(b'~')[0]
        temp = b''
        i = self.min_length
        while temp != b'~':
            temp = data.read(1)
            if temp == b'~':
                return buffer
            elif temp == b'':
                # raise if EOF is reached
                raise SEEDTypeException('Variable string has no terminator')
            buffer += temp
            i = i + 1
        return buffer

    def write(self, data, strict=False):  # @UnusedVariable
        result = self._format_string(data).encode('utf-8')
        if self.max_length and len(result) > self.max_length + 1:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.max_length, self.attribute_name)
            if strict:
                raise SEEDTypeException(msg)
            msg += ' Reducing to %d chars.' % (self.max_length)
            warnings.warn(msg, UserWarning)
            result = result[:self.max_length]
        # MSEED manual p. 30: Character counts for variable length fields do
        # not include the tilde terminator - however this is not valid for
        # minimum sizes - e.g. optional date fields in Blockette 10
        # so we add here the terminator string, and check minimum size below
        result += b'~'
        if len(result) < self.min_length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.min_length, self.attribute_name)
            if strict:
                raise SEEDTypeException(msg)
            delta = self.min_length - len(result)
            msg += ' Adding %d space(s).' % (delta)
            warnings.warn(msg, UserWarning)
            result = b' ' * delta + result
        return result


class Loop(Field):
    """
    A loop over multiple elements.
    """
    def __init__(self, name, index_field, data_fields, **kwargs):
        self.default = False
        Field.__init__(self, None, name, **kwargs)
        # initialize + default values
        if not isinstance(data_fields, list):
            data_fields = [data_fields]
        self.data_fields = data_fields
        self.index_field = to_tag(index_field)
        self.length = 0
        # loop types
        self.repeat_title = kwargs.get('repeat_title', False)
        self.omit_tag = kwargs.get('omit_tag', False)
        self.flat = kwargs.get('flat', False)

    def parse_seed(self, blockette, data):
        """
        """
        try:
            self.length = int(getattr(blockette, self.index_field))
        except Exception:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        # loop over number of entries
        debug = blockette.debug
        blockette.debug = False
        temp = []
        for _i in range(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                field.parse_seed(blockette, data)
                if debug:
                    temp.append(field.data)
        # debug
        if debug:
            if len(temp) > 3:
                print('  LOOP: ... (%d elements) ' % (len(temp)))
            else:
                print('  LOOP: %s' % (temp))
            blockette.debug = debug

    def get_seed(self, blockette):
        """
        """
        try:
            self.length = int(getattr(blockette, self.index_field))
        except Exception:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        # loop over number of entries
        data = b''
        for i in range(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                data += field.get_seed(blockette, i)
        return data

    def get_xml(self, blockette, pos=0):  # @UnusedVariable
        """
        """
        if self.ignore:
            return []
        try:
            self.length = int(getattr(blockette, self.index_field))
        except Exception:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        if self.length == 0 and self.optional:
            return []
        if self.repeat_title:
            # parent tag is repeated over every child tag
            # e.g. <parent><i1/><i2/></parent><parent><i1/><i2/></parent>
            root = Element(self.field_name)
            for _i in range(0, self.length):
                se = SubElement(root, self.field_name)
                # loop over data fields within one entry
                for field in self.data_fields:
                    node = field.get_xml(blockette, _i)
                    se.extend(node)
            return root.getchildren()
        # loop over number of entries
        root = Element(self.field_name)
        for _i in range(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                node = field.get_xml(blockette, _i)
                root.extend(node)
        # format output for requested loop type
        if self.flat:
            # flat loop: one or multiple fields are within one parent tag
            # e.g. <root>item1 item2 item1 item2</root>
            root.text = ' '.join([i.text for i in root.getchildren()])
            root[:] = []
            return [root]
        elif self.omit_tag:
            # loop omitting the parent tag: fields are at the same level
            # e.g. <item1/><item2/><item1/><item2/>
            return root.getchildren()
        else:
            # standard loop
            return [root]

    def parse_xml(self, blockette, xml_doc, pos=0):
        """
        """
        try:
            self.length = int(getattr(blockette, self.index_field))
        except Exception:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        if self.length == 0:
            return
        # loop type
        if self.flat:
            # flat loop: one or multiple fields are within one parent tag
            # e.g. <root>item1 item2 item1 item2</root>
            text = xml_doc.xpath(self.field_name + '/text()')[0].split()
            if not text:
                return
            # loop over number of entries
            for _i in range(0, self.length):
                # loop over data fields within one entry
                for field in self.data_fields:
                    temp = getattr(blockette, field.attribute_name, [])
                    temp.append(field.convert(text.pop(0)))
                    setattr(blockette, field.attribute_name, temp)
            return
        elif self.omit_tag:
            # loop omitting the parent tag: fields are at the same level
            # e.g. <item1/><item2/><item1/><item2/>
            root = Element(self.field_name)
            root.extend(xml_doc)
        elif self.repeat_title:
            # parent tag is repeated over every child tag
            # e.g. <parent><i1/><i2/></parent><parent><i1/><i2/></parent>
            root = Element(self.field_name)
            root.extend(xml_doc.xpath(self.field_name + '/*'))
        else:
            # standard loop
            root = xml_doc.xpath(self.field_name)[pos]
        # loop over number of entries
        for i in range(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                field.parse_xml(blockette, root, i)
