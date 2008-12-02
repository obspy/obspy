# -*- coding: utf-8 -*-

from StringIO import StringIO
from lxml.etree import Element, SubElement

from obspy.xseed.fields import Integer, MultipleLoop, SimpleLoop, Float
from obspy.xseed import utils


class BlocketteLengthException(Exception):
    """Wrong blockette length detected."""
    pass


class Blockette:
    """General blockette handling."""
    
    # default field for each blockette
    fields = []
    default_fields = [
        Integer(1, "Blockette type", 3),
        Integer(2, "Length of blockette", 4, optional=True)
    ]
    
    def __init__(self, *args, **kwargs):
        self.verify = kwargs.get('verify', True)
        self.debug = kwargs.get('debug', False)
        self.strict = kwargs.get('strict', False)
        self.version = kwargs.get('version', 2.4)
        self.record_type = kwargs.get('record_type', None)
        self.record_id = kwargs.get('record_id', None)
        self.parsed = False
        self.blockette_id = "%03d" % self.id
        self.blockette_name = utils.toXMLTag(self.name)
        if self.debug:
            print "----"
            print str(self)
    
    def __str__(self):
        """String representation of this blockette."""
        return self.blockette_id
    
    def parse(self, data, expected_length=0):
        """Parse given data for blockette fields and create attributes."""
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
            temp=data.read(expected_length)
            print '  DATA:', temp
            data.seek(-expected_length,1)
        blockette_fields = self.default_fields + self.fields
        for field in blockette_fields:
            # blockette length reached -> break with warning, because fields 
            # still exist
            if data.tell()-start_pos >= expected_length:
                if not self.strict and not self.verify:
                    break
                if isinstance(field, MultipleLoop) or \
                   isinstance(field, SimpleLoop):
                    break
                msg = "End of blockette " + self.blockette_id + " reached " + \
                      "without parsing all expected fields, here: "+ str(field)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
                break
            # check version
            if field.version and field.version>self.version:
                continue
            if isinstance(field, MultipleLoop):
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
                if field.id==2:
                    expected_length = text
                # set attribute
                setattr(self, field.attribute_name, text)
                # debug
                if self.debug:
                    print('  ' + str(field) + ': '  + str(text))
        # verify or strict tests
        if self.verify or self.strict:
            end_pos = data.tell()
            blockette_length = end_pos-start_pos
            if expected_length != blockette_length:
                msg = 'Wrong size of Blockette %s (%d of %d) in sequence %06d'
                msg = msg  % (self.blockette_id, blockette_length, 
                              expected_length, self.record_id)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
    
    def getXML(self, abbrev_dict={}, show_optional=False):
        """Returns a XML document representing this blockette.
        
        The 'optional' flag will return optional elements too.
        """
        # root element
        doc = Element(self.blockette_name, blockette=self.blockette_id)
        # default field for each blockette
        blockette_fields = self.fields
        for field in blockette_fields:
            # check version
            if field.version and field.version>self.version:
                continue
            # skip if optional
            if not show_optional and field.optional:
                continue
            if isinstance(field, MultipleLoop):
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
                # XML looping element 
                root = SubElement(doc, field.field_name)
                # cycle through all fields
                for i in range(0, number_of_elements):
                    item = SubElement(root, 'item')
                    # cycle through fields
                    for subfield in field.data_fields:
                        result = getattr(self, subfield.attribute_name)[i]
                        if isinstance(subfield, Float):
                            result = subfield.write(result)
                        SubElement(item, 
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
                # root of looping element
                subfield = field.data_field
                elements = []
                for subresult in results:
                    if isinstance(subfield, Float):
                        subresult = subfield.write(subresult)
                    elements.append(unicode(subresult))
                SubElement(doc, field.field_name).text = ' '.join(elements)
            else:
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
                se.text = unicode(result)
        return doc
    
    def getSEEDString(self):
        data = ''
        # cycle trough all fields
        for field in self.fields:
            # check version
            if field.version and field.version>self.version:
                continue
            if isinstance(field, MultipleLoop):
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
                        result = getattr(self, subfield.attribute_name)[i]
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
        return '%03d%04d%s' % (self.id, len(data)+7, data)
