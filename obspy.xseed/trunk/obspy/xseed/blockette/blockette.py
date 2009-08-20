# -*- coding: utf-8 -*-

from StringIO import StringIO
from lxml.etree import Element, SubElement
from obspy.xseed import utils
from obspy.xseed.fields import Integer, Loop, Float


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
        self.verify = kwargs.get('verify', True)
        self.debug = kwargs.get('debug', False)
        self.strict = kwargs.get('strict', False)
        self.record_type = kwargs.get('record_type', None)
        self.record_id = kwargs.get('record_id', None)
        self.blockette_id = "%03d" % self.id
        self.blockette_name = utils.toXMLTag(self.name)
        if self.debug:
            print "----"
            print str(self)
        # set default fields using versions
        XSEED_version = kwargs.get('xseed_version', '1.0')
        SEED_version = kwargs.get('version', 2.4)
        self.blockette_fields = []
        for field in self.fields:
            # Check XML-SEED version
            if field.xseed_version and field.xseed_version != XSEED_version:
                continue
            # Check SEED version
            if field.version and field.version > SEED_version:
                continue
            self.blockette_fields.append(field)


    def __str__(self):
        """
        String representation of this blockette.
        """
        return self.blockette_id

    def parseSEED(self, data, expected_length=0):
        """
        Parse given data for blockette fields and create attributes.
        """
        # convert to stream for test issues
        if isinstance(data, basestring):
            expected_length = len(data)
            data = StringIO(data)
        start_pos = data.tell()
        # debug
        if self.debug:
            print '  DATA:', data.read(expected_length)
            data.seek(-expected_length, 1)
        blockette_fields = self.default_fields + self.blockette_fields
        # loop over all blockette fields
        for field in blockette_fields:
            # blockette length reached -> break with warning, because fields 
            # still exist
            if data.tell() - start_pos >= expected_length:
                if not self.strict and not self.verify:
                    break
                if isinstance(field, Loop):
                    break
                msg = "End of blockette " + self.blockette_id + " reached " + \
                      "without parsing all expected fields, here: " + str(field)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
                break
            # ok lets look into the fields
            if isinstance(field, Loop):
                # we got a loop
                index_field = field.index_field
                # set length
                try:
                    field.length = int(getattr(self, index_field))
                except:
                    raise Exception('Field %s missing' % index_field)
                for subfields in field.getSubFields():
                    for subfield in subfields:
                        text = subfield.read(data)
                        # set attribute
                        temp = getattr(self, subfield.attribute_name, [])
                        temp.append(text)
                        setattr(self, subfield.attribute_name, temp)
                        # debug
                        if self.debug:
                            print('  %s: %s' % (subfield, text))
            else:
                # we got a normal SEED field
                try:
                    text = field.read(data)
                except:
                    if self.strict:
                        raise
                    text = field.default
                if field.id == 2:
                    expected_length = text
                # set attribute
                setattr(self, field.attribute_name, text)
                # debug
                if self.debug:
                    print('  %s: %s' % (field, text))
        # verify or strict tests
        if self.verify or self.strict:
            end_pos = data.tell()
            blockette_length = end_pos - start_pos
            if expected_length != blockette_length:
                msg = 'Wrong size of Blockette %s (%d of %d) in sequence %06d'
                msg = msg % (self.blockette_id, blockette_length,
                              expected_length, self.record_id or 0)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)

    def parseXML(self, xml):
        """
        Reads lxml etree and fills the blockette with the values of it.
        """
        xml = xml.getchildren()
        xml_fields = [_i.tag for _i in xml]
        no_index = False
        for field in self.blockette_fields:
            # Check if field is in the supplied XML tree.
            try:
                index_nr = xml_fields.index(field.attribute_name)
            except:
                if field.optional or isinstance(field, Loop):
                    continue
                if field.optional_if_empty:
                    no_index = True
                else:
                    msg = "Missing attribute %s in Blockette %s"
                    raise Exception(msg % (field.name, self))
            if isinstance(field, Loop) and field.omit_tag:
                xml_text = xml[index_nr].text
                if field.seperate_tags:
                    setattr(self, field.attribute_name, [])
                    # It is necessary to loop over all xml fields.
                    for xml_field in xml:
                        if xml_field.tag != field.attribute_name:
                            continue
                        getattr(self,
                                field.attribute_name).append(xml_field.text)
                if not field.seperate_tags:
                    setattr(self, field.attribute_name, xml_text.split(' '))
            elif isinstance(field, Loop) and field.flat:
                xml_field = xml[index_nr].getchildren()
                if len(xml_field) == 0:
                    for data_field in field.data_fields:
                        setattr(self, data_field.attribute_name, 0)
                    continue
                if len(xml_field) % len(field.data_fields) != 0:
                    msg = 'Ooops...something went wrong!'
                    raise BlocketteParserException(msg)
                # Set lists.
                for data_field in field.data_fields:
                    setattr(self, data_field.attribute_name, [])
                # Fill the lists.
                index_nr = 0
                for _i in xrange(len(xml_field)):
                    getattr(self, field.data_fields[index_nr % \
                        len(xml_field)]).append(xml_field[index_nr])
                    index_nr += 1
            elif isinstance(field, Loop):
                # all other loops
                xml_childs = xml[index_nr].getchildren()
                xml_child_fields = [_i.tag for _i in xml_childs]
                # Loop over all data fields.
                for subfield in field.data_fields:
                    id = subfield.attribute_name
                    setattr(self, id, [])
                    # Check if all necessary attributes are set.
                    if not id in xml_child_fields:
                        if subfield.optional:
                            continue
                        else:
                            msg = "Missing attribute %s in Blockette %s"
                            raise Exception(msg % (subfield.name, self))
                    for child_field in xml_childs:
                        if not child_field.tag == id:
                            continue
                        xml_text = child_field.text or child_field.default
                        getattr(self, id).append(xml_text)
            else:
                if field.optional_if_empty and no_index:
                    setattr(self, field.attribute_name, '')
                    continue
                xml_text = xml[index_nr].text or field.default
                setattr(self, field.attribute_name, xml_text)

    def getXML(self, show_optional=False):
        """
        Returns a XML document representing this blockette.
        """
        # root element
        doc = Element(self.blockette_name, blockette=self.blockette_id)
        # default field for each blockette
        for field in self.blockette_fields:
            # skip if optional
            if not show_optional and field.optional:
                continue
            # skip if ignore
            if field.ignore:
                continue
            if isinstance(field, Loop):
                # we got a loop
                try:
                    number_of_elements = int(getattr(self, field.index_field))
                except:
                    msg = "Missing attribute %s in Blockette %s"
                    raise Exception(msg % (field.index_field, self))

                if field.flat:
                    # get number of entries
                    if number_of_elements == 0:
                        # Write empty tag and continue.
                        SubElement(doc, field.field_name).text = ''
                        continue
                    # test if attributes of subfields are set
                    for subfield in field.data_fields:
                        if subfield.ignore:
                            continue
                        if not hasattr(self, subfield.attribute_name):
                            msg = "Missing attribute %s in Blockette %s"
                            raise Exception(msg % (subfield.name, self))
                    # XML looping element 
                    elements = []
                    # cycle through all fields
                    for i in xrange(0, number_of_elements):
                        # cycle through fields
                        for subfield in field.data_fields:
                            if subfield.ignore:
                                continue
                            result = getattr(self, subfield.attribute_name)[i]
                            if isinstance(subfield, Float):
                                result = subfield.write(result)
                            elements.append(unicode(result))
                    SubElement(doc, field.field_name).text = ' '.join(elements)
                else:
                    if number_of_elements == 0:
                        continue
                    # test if attributes of subfields are set
                    for subfield in field.data_fields:
                        if subfield.ignore:
                            continue
                        if not hasattr(self, subfield.attribute_name):
                            msg = "Attribute %s in Blockette %s does not exist!"
                            msg = msg % (subfield.name, self.blockette_id)
                            raise Exception(msg)
                    if field.omit_tag:
                        root = doc
                    elif not field.repeat_title:
                        # XML looping element
                        root = SubElement(doc, field.field_name)
                    # cycle through all fields
                    for i in xrange(0, number_of_elements):
                        if field.repeat_title:
                            root = SubElement(doc, field.field_name)
                        # cycle through fields
                        for subfield in field.data_fields:
                            if subfield.ignore:
                                continue
                            result = getattr(self, subfield.attribute_name)[i]
                            if isinstance(subfield, Float):
                                result = subfield.write(result)
                            SubElement(root,
                                subfield.field_name).text = unicode(result)
            else:
                try:
                    result = getattr(self, field.attribute_name)
                except:
                    if self.strict:
                        msg = "Missing attribute %s in Blockette %s"
                        raise Exception(msg % (field.name, self))
                    result = field.default
                # optional if empty
                if field.optional_if_empty and len(result) == 0:
                    continue
                # reformat float
                if isinstance(field, Float):
                    result = field.write(result)
                # set XML string
                se = SubElement(doc, field.field_name)
                se.text = unicode(result).strip()
        return doc

    def getSEED(self):
        """
        Converts the blockette to a valid SEED string and returns it.
        """
        data = ''
        # loop over all blockette fields
        for field in self.blockette_fields:
            # ok lets look into the fields
            if isinstance(field, Loop):
                # we got a loop
                try:
                    number_of_elements = int(getattr(self, field.index_field))
                except:
                    msg = "Missing attribute %s in Blockette %s"
                    raise Exception(msg % (field.index_field, self))
                for i in xrange(0, number_of_elements):
                    for subfield in field.data_fields:
                        try:
                            result = getattr(self, subfield.attribute_name)[i]
                        except:
                            msg = "Missing attribute %s in Blockette %s"
                            raise Exception(msg % (subfield.name, self))
                        data += subfield.write(result)
            else:
                # we got a normal SEED field
                try:
                    result = getattr(self, field.attribute_name)
                except:
                    if self.strict:
                        msg = "Missing attribute %s in Blockette %s"
                        raise Exception(msg % (field.name, self))
                    result = field.default
                data += field.write(result)
        # add blockette id and length
        return '%03d%04d%s' % (self.id, len(data) + 7, data)
