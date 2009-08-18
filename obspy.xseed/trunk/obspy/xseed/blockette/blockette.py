# -*- coding: utf-8 -*-

from StringIO import StringIO
from lxml.etree import Element, SubElement
from obspy.xseed import utils
from obspy.xseed.fields import Integer, MultipleLoop, SimpleLoop, Float, \
    MultipleFlatLoop, VariableString, FixedString


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



class Blockette:
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
        self.version = kwargs.get('version', 2.4)
        self.record_type = kwargs.get('record_type', None)
        self.record_id = kwargs.get('record_id', None)
        self.parsed = False
        self.blockette_id = "%03d" % self.id
        self.blockette_name = utils.toXMLTag(self.name)
        self.xseed_version = kwargs.get('xseed_version', '1.0')
        if self.debug:
            print "----"
            print str(self)

    def __str__(self):
        """
        String representation of this blockette.
        """
        return self.blockette_id

    def parse(self, data, expected_length=0):
        """
        Parse given data for blockette fields and create attributes.
        """
        # parse only once per Blockette
        if self.parsed:
            raise Exception('Blockette should be parsed only once.')
        self.parsed = True
        # convert to stream for test issues
        if isinstance(data, basestring):
            expected_length = len(data)
            data = StringIO(data)
        start_pos = data.tell()
        if self.debug:
            temp = data.read(expected_length)
            print '  DATA:', temp
            data.seek(-expected_length, 1)
        blockette_fields = self.default_fields + self.fields
        for field in blockette_fields:
            # Check for wanted xseed_version
            if field.xseed_version and \
               field.xseed_version != self.xseed_version:
                continue
            # blockette length reached -> break with warning, because fields 
            # still exist
            if data.tell() - start_pos >= expected_length:
                if not self.strict and not self.verify:
                    break
                if isinstance(field, MultipleLoop) or \
                   isinstance(field, SimpleLoop):
                    break
                msg = "End of blockette " + self.blockette_id + " reached " + \
                      "without parsing all expected fields, here: " + str(field)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
                break
            # check version
            if field.version and field.version > self.version:
                continue
            if isinstance(field, MultipleLoop) or \
               isinstance(field, MultipleFlatLoop):
                index_field = field.index_field
                # test if index attribute is set
                if not hasattr(self, index_field):
                    raise Exception('Field %s missing' % index_field)
                # set length
                field.length = int(getattr(self, index_field))
                for subfields in field.getSubFields():
                    for subfield in subfields:
                        text = subfield.read(data)
                        # set attribute
                        temp = getattr(self, subfield.attribute_name, [])
                        temp.append(text)
                        setattr(self, subfield.attribute_name, temp)
                        # debug
                        if self.debug:
                            print('    ' + str(subfield) + ': ' + str(text))
            elif isinstance(field, SimpleLoop):
                index_field = field.index_field
                # test if index attribute is set
                if not hasattr(self, index_field):
                    raise Exception('Field %s missing' % index_field)
                # set length
                field.length = int(getattr(self, index_field))
                # set attributes
                for subfield in field.getSubFields():
                    text = subfield.read(data)
                    # attribute
                    temp = getattr(self, subfield.attribute_name, [])
                    temp.append(text)
                    setattr(self, subfield.attribute_name, temp)
                    # debug
                    if self.debug:
                        print('    ' + str(subfield) + ': ' + str(text))
            else:
                text = field.read(data)
                if field.id == 2:
                    expected_length = text
                # set attribute
                setattr(self, field.attribute_name, text)
                # debug
                if self.debug:
                    print('  ' + str(field) + ': ' + str(text))
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
        for field in self.fields:
            # Check for wanted xseed_version
            if field.xseed_version and \
               field.xseed_version != self.xseed_version:
                continue
            # Check if field is in the supplied XML tree.
            try:
                index_nr = xml_fields.index(field.attribute_name)
            except:
                if field.optional or isinstance(field, MultipleLoop):
                    continue
                if field.optional_if_empty:
                    no_index = True
                else:
                    msg = 'Not all necessary fields found for Blockette %i'\
                                    % self.id
                    raise BlocketteParserException(msg)
            # If field is a Float or Integer convert the XML to a float or int
            # and write it into the blockette.
            if isinstance(field, Float):
                xml_text = xml[index_nr].text
                if not xml_text:
                    xml_text = 0
                setattr(self, field.attribute_name, float(xml_text))
            if isinstance(field, Integer):
                xml_text = xml[index_nr].text
                if not xml_text:
                    xml_text = 0
                setattr(self, field.attribute_name, int(xml_text))
            # If its a string leave it being a string.
            if isinstance(field, VariableString) or \
                                        isinstance(field, FixedString):
                if field.optional_if_empty and no_index:
                    setattr(self, field.attribute_name, '')
                    continue
                xml_text = xml[index_nr].text
                if not xml_text:
                    xml_text = ''
                setattr(self, field.attribute_name, xml_text)
            if isinstance(field, MultipleFlatLoop):
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
            elif isinstance(field, MultipleLoop):
                xml_childs = xml[index_nr].getchildren()
                xml_child_fields = [_i.tag for _i in xml_childs]
                # Loop over all data fields.
                for sub_field in field.data_fields:
                    setattr(self, sub_field.attribute_name, [])
                    # Check if all necessary attributes are set.
                    if not sub_field.attribute_name in xml_child_fields:
                        if sub_field.optional:
                            continue
                        else:
                            import pdb;pdb.set_trace()
                            msg = 'Not all necessary fields found for ' + \
                                  'Blockette %i' % self.id
                            raise BlocketteParserException(msg)
                    for child_field in xml_childs:
                        if not child_field.tag == sub_field.attribute_name:
                            continue
                        # Check whether it's a Float, an Integer or a String.
                        if isinstance(sub_field, Float):
                            xml_text = child_field.text
                            if not xml_text:
                                xml_text = 0
                            getattr(self, sub_field.attribute_name).append(\
                                    float(xml_text))
                        elif isinstance(sub_field, Integer):
                            xml_text = child_field.text
                            if not xml_text:
                                xml_text = 0
                            getattr(self, sub_field.attribute_name).append(\
                                    int(xml_text))
                        else:
                            xml_text = child_field.text
                            if not xml_text:
                                xml_text = ''
                            getattr(self, sub_field.attribute_name).append(\
                                    xml_text)
            if isinstance(field, SimpleLoop):
                xml_text = xml[index_nr].text
                if field.seperate_tags:
                    setattr(self, field.attribute_name, [])
                    # It is necessary to loop over all xml fields.
                    for xml_field in xml:
                        if xml_field.tag == field.attribute_name:
                            getattr(self,
                                field.attribute_name).append(xml_field.text)
                if not field.seperate_tags:
                    setattr(self, field.attribute_name, xml_text.split(' '))

    def getXML(self, show_optional=False):
        """
        Returns a XML document representing this blockette.
        """
        # root element
        doc = Element(self.blockette_name, blockette=self.blockette_id)
        # default field for each blockette
        blockette_fields = self.fields
        for field in blockette_fields:
            # check version
            if field.version and field.version > self.version:
                continue
            # Check for wanted xseed_version
            if field.xseed_version and \
               field.xseed_version != self.xseed_version:
                continue
            # skip if optional
            if not show_optional and field.optional:
                continue
            if isinstance(field, MultipleFlatLoop):
                # test if index attribute is set
                if not hasattr(self, field.index_field):
                    msg = "Attribute %s in Blockette %s does not exist!"
                    msg = msg % (field.index_field, self.blockette_id)
                    raise Exception(msg)
                # get number of entries
                number_of_elements = int(getattr(self, field.index_field))
                if number_of_elements == 0:
                    # Write empty tag and continue.
                    SubElement(doc, field.field_name).text = ''
                    continue
                # test if attributes of subfields are set
                for subfield in field.data_fields:
                    if subfield.ignore:
                        continue
                    if not hasattr(self, subfield.attribute_name):
                        msg = "Attribute %s in Blockette %s does not exist!"
                        msg = msg % (subfield.name, self.blockette_id)
                        raise Exception(msg)
                # XML looping element 
                elements = []
                # cycle through all fields
                for i in range(0, number_of_elements):
                    # cycle through fields
                    for subfield in field.data_fields:
                        if subfield.ignore:
                            continue
                        result = getattr(self, subfield.attribute_name)[i]
                        if isinstance(subfield, Float):
                            result = subfield.write(result)
                        elements.append(unicode(result))
                SubElement(doc, field.field_name).text = ' '.join(elements)
            elif isinstance(field, MultipleLoop):
                # test if index attribute is set
                if not hasattr(self, field.index_field):
                    msg = "Attribute %s in Blockette %s does not exist!"
                    msg = msg % (field.index_field, self.blockette_id)
                    raise Exception(msg)
                # get number of entries
                number_of_elements = int(getattr(self, field.index_field))
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
                if not field.repeat_title:
                    # XML looping element
                    root = SubElement(doc, field.field_name)
                # cycle through all fields
                for i in range(0, number_of_elements):
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
            elif isinstance(field, SimpleLoop):
                # test if index attribute is set
                if not hasattr(self, field.index_field):
                    msg = "Attribute %s in Blockette %s does not exist!"
                    msg = msg % (field.index_field, self.blockette_id)
                    raise Exception(msg)
                # get number of entries
                number_of_elements = int(getattr(self, field.index_field))
                if number_of_elements == 0:
                    continue
                # check if attribute exists
                if not hasattr(self, field.attribute_name):
                    msg = "Attribute %s in Blockette %s does not exist!"
                    msg = msg % (field.attribute_name, self.blockette_id)
                    raise Exception(msg)
                results = getattr(self, field.attribute_name)
                subfield = field.data_field
                if not field.seperate_tags:
                    # root of looping element
                    elements = []
                    for subresult in results:
                        if isinstance(subfield, Float):
                            subresult = subfield.write(subresult)
                        #SubElement(doc, field.field_name).text = unicode(subresult)
                        elements.append(unicode(subresult))
                    SubElement(doc, field.field_name).text = ' '.join(elements)
                else:
                    # root of looping element
                    for subresult in results:
                        SubElement(doc, field.field_name).text = subresult
            else:
                if isinstance(field, VariableString):
                    if field.optional_if_empty and \
                            len(getattr(self, field.attribute_name)) == 0:
                        continue
                # check if ignored
                if field.ignore:
                    continue
                # check if attribute exists
                if not hasattr(self, field.attribute_name):
                    if self.strict:
                        msg = "Missing attribute %s in Blockette %s" % \
                              (field.attribute_name, self.blockette_id)
                        raise Exception(msg)
                    result = field.default
                else:
                    result = getattr(self, field.attribute_name)
                if isinstance(field, Float):
                    result = field.write(result)
                # set XML string
                se = SubElement(doc, field.field_name)
                se.text = unicode(result).strip()
        return doc

    def getSEEDString(self):
        """
        Converts the blockette to a valid SEED string and returns it.
        """
        data = ''
        # cycle trough all fields
        for field in self.fields:
            # check version
            if field.version and field.version > self.version:
                print 'ACHTUNG!'
                continue
            # Check for wanted xseed_version
            if field.xseed_version and \
               field.xseed_version != self.xseed_version:
                continue
            if isinstance(field, MultipleLoop) or \
               isinstance(field, MultipleFlatLoop):
                # test if index attribute is set
                if not hasattr(self, field.index_field):
                    msg = "Attribute %s in Blockette %s does not exist!"
                    msg = msg % (field.index_field, self.blockette_id)
                    raise Exception(msg)
                # get number of entries
                number_of_elements = int(getattr(self, field.index_field))
                if number_of_elements == 0:
                    continue
                # test if attributes of subfields are set
                for subfield in field.data_fields:
                    if not hasattr(self, subfield.attribute_name):
                        msg = "Attribute %s in Blockette %s does not exist!"
                        msg = msg % (subfield.name, self.blockette_id)
                        raise Exception(msg)
                # cycle through all fields
                for i in range(0, number_of_elements):
                    # cycle through fields
                    for subfield in field.data_fields:
                        try:
                            result = getattr(self, subfield.attribute_name)[i]
                        except:
                            import pdb;pdb.set_trace()
                        data = data + subfield.write(result)
            elif isinstance(field, SimpleLoop):
                # test if index attribute is set
                if not hasattr(self, field.index_field):
                    msg = "Attribute %s in Blockette %s does not exist!"
                    msg = msg % (field.index_field, self.blockette_id)
                    raise Exception(msg)
                # get number of entries
                number_of_elements = int(getattr(self, field.index_field))
                if number_of_elements == 0:
                    continue
                # check if attribute exists
                if not hasattr(self, field.attribute_name):
                    msg = "Attribute %s in Blockette %s does not exist!"
                    msg = msg % (field.attribute_name, self.blockette_id)
                    raise Exception(msg)
                results = getattr(self, field.attribute_name)
                subfield = field.data_field
                # root of looping element
                for result in results:
                    data = data + subfield.write(result)
            else:
                # check if attribute exists
                if not hasattr(self, field.attribute_name):
                    if self.strict:
                        msg = "Missing attribute %s in Blockette %s"
                        msg = msg % (field.attribute_name, self.blockette_id)
                        raise Exception(msg)
                    result = field.default
                else:
                    result = getattr(self, field.attribute_name)
                data = data + field.write(result)
        # add blockette id and length
        return '%03d%04d%s' % (self.id, len(data) + 7, data)
