# -*- coding: utf-8 -*-

from StringIO import StringIO
from lxml.etree import Element, SubElement

from obspy.seed.fields import Integer, MultipleLoop, SimpleLoop


class BlocketteLengthException(Exception):
    """Wrong blockette length detected."""
    pass


class Blockette:
    """General blockette handling."""
    
    def __init__(self, *args, **kwargs):
        self.debug = kwargs.get('debug', False)
        self.verify = kwargs.get('verify', True)
        self.strict = kwargs.get('strict', False)
        self.version = kwargs.get('version', 2.4)
        self.parsed = False
        if self.debug:
            print "----"
            print str(self)
    
    def __str__(self):
        """String representation of this blockette."""
        return "B%03d" % self.id
    
    def parse(self, data, expected_length=0):
        """Parse given data for blockette fields and create attributes."""
        # parse only once per Blockette
        if self.parsed:
            raise Exception('Blockette should be parsed only once.')
        # for test issues
        if isinstance(data, basestring):
            expected_length = len(data)
            data = StringIO(data)
        start_pos = data.tell()
        if self.debug:
            temp=data.read(expected_length)
            print '  DATA:', temp
            data.seek(-expected_length,1)
        blockette_id = str(self)
        blockette_name = self.name.title().replace(' ','')
        doc = Element(blockette_name, id=blockette_id)
        blockette_fields = [Integer(1, "Blockette type", 3),
                            Integer(2, "Length of blockette", 4)]
        blockette_fields.extend(self.fields)
        for field in blockette_fields:
            # blockette length reached -> break with warning, because fields 
            # still exist
            if data.tell()-start_pos >= expected_length:
                if not self.strict and not self.verify:
                    break
                if isinstance(field, MultipleLoop) or \
                   isinstance(field, SimpleLoop):
                    break
                msg = "End of blockette " + blockette_id + " reached " + \
                      "without parsing all expected fields, here: "+ str(field)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
                break
            # check version
            if field.version and field.version>self.version:
                break
            field_name = field.name.title().replace(' ','')
            attribute_name = field.name.lower().replace(' ','_')
            if isinstance(field, MultipleLoop):
                index_field = field.index_field
                # test if index attribute is set
                if not hasattr(self, index_field):
                    raise Exception('Field %s missing' % index_field)
                # set length
                field.length = int(getattr(self, index_field))
                loop_fields = field.read(data)
                # set XML string
                root = SubElement(doc, field_name)
                for subfields in loop_fields:
                    item = SubElement(root, 'item')
                    for subfield in subfields:
                        subresult = subfield.read(data)
                        subname = subfield.name.title().replace(' ','')
                        # XML string
                        text = unicode(subresult)
                        SubElement(item, subname).text = text
                        # set attribute
                        attribute_name = subfield.name.lower().replace(' ','_')
                        temp = getattr(self, attribute_name, [])
                        temp.append(subresult)
                        setattr(self, attribute_name, temp)
                        # debug
                        if self.debug:
                            print('    ' + str(subfield) + ': ' + text)
            elif isinstance(field, SimpleLoop):
                index_field = field.index_field
                # test if index attribute is set
                if not hasattr(self, index_field):
                    raise Exception('Field %s missing' % index_field)
                # set length
                field.length = int(getattr(self, index_field))
                loop_fields = field.read(data)
                # set XML string and attributes
                root = SubElement(doc, field_name)
                for subfield in loop_fields:
                    subresult = subfield.read(data)
                    # XML string
                    text = unicode(subresult)
                    SubElement(root, 'item').text = text
                    # attribute
                    temp = getattr(self, attribute_name, [])
                    temp.append(subresult)
                    setattr(self, attribute_name, temp)
                    # debug
                    if self.debug:
                        print('    ' + str(subfield) + ': ' + text)
            else:
                result = field.read(data)
                if field.id==2:
                    expected_length = result
                # set attribute
                setattr(self, attribute_name, result)
                # set XML string
                text = unicode(result)
                SubElement(doc, field_name, id=str(field)).text = text
                # debug
                if self.debug:
                    print('  ' + str(field) + ': '  + text)
        end_pos = data.tell()
        if self.verify or self.strict:
            blockette_length = end_pos-start_pos
            if expected_length != blockette_length:
                msg = 'Wrong size of Blockette %s (%d of %d)' % \
                      (blockette_id, blockette_length, expected_length)
                if self.strict:
                    raise BlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
        return doc
