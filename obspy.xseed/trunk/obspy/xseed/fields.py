# -*- coding: utf-8 -*-

from lxml.etree import Element, SubElement
from obspy.core.util import formatScientific
from obspy.xseed import utils
import re


class SEEDTypeException(Exception):
    pass


class Field(object):
    """
    General SEED field.
    """
    def __init__(self, id, name, *args, **kwargs):
        # default
        self.id = id
        self.name = name
        self.xseed_version = kwargs.get('xseed_version', None)
        self.optional_if_empty = kwargs.get('optional_if_empty', False)
        self.version = kwargs.get('version', None)
        self.mask = kwargs.get('mask', None)
        if self.id:
            self.field_id = "F%02d" % self.id
        else:
            self.field_id = None
        self.field_name = utils.toXMLTag(self.name)
        self.attribute_name = utils.toAttribute(self.name)
        # options
        self.optional = kwargs.get('optional', False)
        self.ignore = kwargs.get('ignore', False)
        self.strict = kwargs.get('strict', False)

    def __str__(self):
        if self.id:
            return "F%02d" % self.id

    def _formatString(self, s, flags=None):
        """
        Using SEED specific flags to format strings.
        
        This method is partly adopted from fseed.py, the SEED builder for 
        SeisComP written by Andres Heinloo, GFZ Potsdam in 2005.
        """
        if flags and 'T' in flags:
            dt = utils.Iso2DateTime(s)
            return utils.DateTime2String(dt)
        sn = str(s).strip()
        if not flags:
            return sn
        rx_list = []
        if 'U' in flags:
            rx_list.append("[A-Z]")
        if 'L' in flags:
            rx_list.append("[a-z]")
        if 'N' in flags:
            rx_list.append("[0-9]")
        if 'P' in flags:
            rx_list.append("[^A-Za-z0-9 ]")
        if 'S' in flags:
            rx_list.append(" ")
        if '_' in flags:
            rx_list.append("_")
        if 'U' in flags and 'L' not in flags:
            sn = sn.upper()
        elif 'L' in flags and 'U' not in flags:
            sn = sn.lower()
        if 'S' in flags and 'X' not in flags:
            sn = sn.replace("_", " ")
        elif 'X' in flags and 'S' not in flags:
            sn = sn.replace(" ", "_")
        rx = "|".join(rx_list)
        sn = "".join(re.findall(rx, sn))
        if re.match("(" + rx + ")*$", sn) == None:
            msg = "Can't convert string %s with flags %s" % (s, flags)
            raise SEEDTypeException(msg)
        return sn

    def parseSEED(self, blockette, data):
        """
        """
        try:
            text = self.read(data)
        except:
            if blockette.strict:
                raise
            # default value if not set
            text = self.default
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

    def getSEED(self, blockette, pos=0):
        """
        """
        try:
            result = getattr(blockette, self.attribute_name)
        except:
            if blockette.strict:
                msg = "Missing attribute %s in Blockette %s"
                raise Exception(msg % (self.name, blockette))
            result = self.default
        # watch for multiple entries
        if isinstance(result, list):
            result = result[pos]
        return self.write(result)

    def getXML(self, blockette, pos=0):
        """
        """
        if self.ignore:
            return []
        try:
            result = getattr(blockette, self.attribute_name)
        except:
            if blockette.strict:
                msg = "Missing attribute %s in Blockette %s"
                raise Exception(msg % (self.name, blockette))
            result = self.default
        # watch for multiple entries
        if isinstance(result, list):
            result = result[pos]
        # optional if empty
        if self.optional_if_empty and len(result) == 0:
            return []
        # reformat float
        if isinstance(self, Float):
            result = self.write(result)
        # create XML element
        node = Element(self.field_name)
        node.text = unicode(result).strip()
        return [node]

    def parseXML(self, blockette, xml_doc, pos=0):
        """
        """
#        if field.optional_if_empty and no_index:
#            setattr(self, field.attribute_name, '')
#            continue
        try:
            text = xml_doc.xpath(self.attribute_name + "/text()")[pos]
        except:
            return
        # check if already exists
        if hasattr(blockette, self.attribute_name):
            temp = getattr(blockette, self.attribute_name)
            if not isinstance(temp, list):
                temp = [temp]
            temp.append(text)
            text = temp
        setattr(blockette, self.attribute_name, text)


class Integer(Field):
    """
    An integer field.
    """
    def __init__(self, id, name, length, **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.default = 0

    def read(self, data):
        temp = data.read(self.length)
        try:
            temp = int(temp)
        except:
            msg = "No integer value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        return temp

    def write(self, data):
        format_str = "%%0%dd" % self.length
        try:
            temp = int(data)
        except:
            msg = "No integer value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        result = format_str % temp
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class Float(Field):
    """
    A float number with a fixed length and output mask.
    """
    def __init__(self, id, name, length, **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.default = 0
        if not self.mask:
            msg = "Float field %s requires a data mask." % self.field_name
            raise SEEDTypeException(msg)

    def read(self, data):
        temp = data.read(self.length)
        try:
            temp = float(temp)
        except:
            msg = "No float value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        return temp

    def write(self, data):
        format_str = "%%0%ds" % self.length
        try:
            temp = float(data)
        except:
            msg = "No float value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        # special format for exponential output
        result = format_str % (self.mask % temp)
        if 'E' in self.mask or 'e' in self.mask:
            result = formatScientific(result.upper())
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class FixedString(Field):
    """
    An string field with a fixed width.
    """
    def __init__(self, id, name, length, flags='', **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.flags = flags
        self.default = ' ' * length

    def read(self, data):
        return data.read(self.length).strip()

    def write(self, data):
        # Leave fixed length alphanumeric fields left justified (no leading 
        # spaces), and pad them with spaces (after the field’s contents).
        format_str = "%%-%ds" % self.length
        result = format_str % self._formatString(data, self.flags)
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class VariableString(Field):
    """
    Variable length ASCII string, ending with a tilde: ~ (ASCII 126).
    
    Variable length fields cannot have leading or trailing spaces. Character 
    counts for variable length fields do not include the tilde terminator. 
    """
    def __init__(self, id, name, min_length=0, max_length=None, flags=None,
                 **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.flags = flags
        self.default = ' ' * min_length

    def read(self, data):
        data = self._read(data)
        # datetime ?
        if self.flags and 'T' in self.flags:
            # convert to ISO 8601 time strings
            dt = utils.String2DateTime(data)
            return utils.DateTime2Iso(dt)
        else:
            return data

    def _read(self, data):
        buffer = ''
        if self.min_length:
            buffer = data.read(self.min_length)
            if '~' in buffer:
                return buffer.split('~')[0]
        temp = ''
        i = self.min_length
        while temp != '~':
            temp = data.read(1)
            if temp == '~':
                return buffer
            buffer += temp
            i = i + 1
            if self.max_length and i > self.max_length:
                return buffer
        return buffer

    def write(self, data):
        result = self._formatString(data, self.flags) + '~'
        if self.max_length and len(result) > self.max_length + 1:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        if len(result) < self.min_length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class Loop(Field):
    """
    A loop over multiple elements.
    """
    def __init__(self, name, index_field, data_fields, **kwargs):
        Field.__init__(self, None, name, **kwargs)
        # initialize + default values
        if not isinstance(data_fields, list):
            data_fields = [data_fields]
        self.data_fields = data_fields
        self.index_field = utils.toAttribute(index_field)
        self.length = 0
        # loop types
        self.repeat_title = kwargs.get('repeat_title', False)
        self.seperate_tags = kwargs.get('seperate_tags', False)
        self.omit_tag = kwargs.get('omit_tag', False)
        self.flat = kwargs.get('flat', False)

    def parseSEED(self, blockette, data):
        """
        """
        try:
            self.length = int(getattr(blockette, self.index_field))
        except:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        # loop over number of entries
        for _i in xrange(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                field.parseSEED(blockette, data)

    def getSEED(self, blockette):
        """
        """
        try:
            self.length = int(getattr(blockette, self.index_field))
        except:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        # loop over number of entries
        data = ''
        for i in xrange(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                data += field.getSEED(blockette, i)
        return data

    def getXML(self, blockette, pos=0):
        """
        """
        if self.ignore:
            return []
        try:
            self.length = int(getattr(blockette, self.index_field))
        except:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        if self.length == 0 and self.optional:
            return []
        # loop over number of entries
        root = Element(self.field_name)
        for _i in xrange(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                node = field.getXML(blockette, _i)
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
        elif self.repeat_title:
            # parent tag is repeated over every child tag
            # e.g. <parent><item1/></parent><parent><item2/></parent>
            children = root.getchildren()
            root = Element(self.field_name)
            for child in children:
                se = SubElement(root, self.field_name)
                se.append(child)
            return root.getchildren()
        elif self.seperate_tags:
            # XXX
            import pdb;pdb.set_trace()
        else:
            # standard loop
            return [root]

    def parseXML(self, blockette, xml_doc, pos=0):
        try:
            self.length = int(getattr(blockette, self.index_field))
        except:
            msg = "Missing attribute %s in Blockette %s"
            raise Exception(msg % (self.index_field, blockette))
        if self.length == 0:
            return
        # loop type
        if self.flat:
            # flat loop: one or multiple fields are within one parent tag
            # e.g. <root>item1 item2 item1 item2</root>
            text = xml_doc.xpath(self.attribute_name + '/text()')[0].split()
            if not text:
                return
            # loop over number of entries
            for _i in xrange(0, self.length):
                # loop over data fields within one entry
                for field in self.data_fields:
                    temp = getattr(blockette, field.attribute_name, [])
                    temp.append(text.pop(0))
                    setattr(blockette, field.attribute_name, temp)
            return
        elif self.omit_tag:
            # loop omitting the parent tag: fields are at the same level
            # e.g. <item1/><item2/><item1/><item2/>
            root = Element(self.attribute_name)
            root.extend(xml_doc)
        elif self.repeat_title:
            # parent tag is repeated over every child tag
            # e.g. <parent><item1/></parent><parent><item2/></parent>
            root = Element(self.attribute_name)
            root.extend(xml_doc.xpath(self.attribute_name + '/*'))
        elif self.seperate_tags:
            # parent tag is repeated over every child tag
            # e.g. <parent><i1/><i2/></parent><parent><i1/><i2/></parent>
            import pdb;pdb.set_trace()
        else:
            # standard loop
            root = xml_doc.xpath(self.attribute_name)[pos]

        # loop over number of entries
        for i in xrange(0, self.length):
            # loop over data fields within one entry
            for field in self.data_fields:
                field.parseXML(blockette, root, i)
