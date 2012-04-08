# -*- coding: utf-8 -*-
from lxml import etree as lxml_etree
from xml.etree import ElementTree as xml_etree
from obspy.core.util.xmlwrapper import XMLParser, tostring
import StringIO
import os
import unittest


XML_DOC = """<?xml version="1.0"?>
<arclink>
  <request args="" ready="true" size="531" type="ROUTING">
    <volume dcid="GFZ" size="531" status="OK">
      <line content="2009,8,24,0,20,2 2009,8,24,0,20,34 BW RJOB . ."
            message="" size="0" status="OK" />
      <line content="2011,8,24,0,20,2 2011,8,24,0,20,34 BW RJOB . ."
            message="" size="12" status="ERROR" />
    </volume>
  </request>
  <request>
    <test muh="kuh" />
  </request>
</arclink>
"""


class XMLWrapperTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.xmlwrapper
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.iris_xml = os.path.join(path, 'iris_events.xml')
        self.neries_xml = os.path.join(path, 'neries_events.xml')

    def test_init(self):
        """
        Tests the __init__ method of the XMLParser object.
        """
        # parser accepts
        # 1 - filenames
        XMLParser(self.iris_xml)
        # 2 - XML strings
        data = XML_DOC
        XMLParser(data)
        # 3 - file like objects
        fh = open(self.iris_xml, 'rt')
        XMLParser(fh)
        fh.close()
        # 4 - StringIO
        data = StringIO.StringIO(XML_DOC)
        XMLParser(data)
        # 5 - with xml parsed XML documents
        xml_doc = xml_etree.parse(self.iris_xml)
        XMLParser(xml_doc)
        # 6 - with lxml parsed XML documents
        xml_doc = lxml_etree.parse(self.iris_xml)
        XMLParser(xml_doc)

    def test_xpath(self):
        """
        Tests the xpath method of the XMLParser object.
        """
        parser = XMLParser(XML_DOC)
        # 1st level
        q = parser.xpath('notexisting')
        self.assertEquals([e.tag for e in q], [])
        q = parser.xpath('request')
        self.assertEquals([e.tag for e in q], ['request', 'request'])
        q = parser.xpath('/request')
        self.assertEquals([e.tag for e in q], ['request', 'request'])
        q = parser.xpath('*')
        self.assertEquals([e.tag for e in q], ['request', 'request'])
        q = parser.xpath('/*')
        self.assertEquals([e.tag for e in q], ['request', 'request'])
        # 2nd level
        q = parser.xpath('*/*')
        self.assertEquals([e.tag for e in q], ['volume', 'test'])
        q = parser.xpath('/*/*')
        self.assertEquals([e.tag for e in q], ['volume', 'test'])
        q = parser.xpath('*/volume')
        self.assertEquals([e.tag for e in q], ['volume'])
        q = parser.xpath('request/*')
        self.assertEquals([e.tag for e in q], ['volume', 'test'])
        q = parser.xpath('request/volume')
        self.assertEquals([e.tag for e in q], ['volume'])
        q = parser.xpath('/request/volume')
        self.assertEquals([e.tag for e in q], ['volume'])
        # 3rd level
        q = parser.xpath('*/*/*')
        self.assertEquals([e.tag for e in q], ['line', 'line'])
        q = parser.xpath('/request/test/doesnotexist')
        self.assertEquals([e.tag for e in q], [])
        # element selector (first element starts with 1)
        q = parser.xpath('/*/*/*[2]')
        self.assertEquals([e.tag for e in q], ['line'])
        q = parser.xpath('/*/*/*[100]')
        self.assertEquals([e.tag for e in q], [])

    def test_getRootNamespace(self):
        """
        Tests for XMLParser._getRootNamespace
        """
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.2")
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.2")
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.0")

    def test_getElementNamespace(self):
        """
        Tests for XMLParser._getElementNamespace
        """
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/bed/1.2")
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/bed/1.2")
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # checking sub elements
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/bed/1.2")
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/bed/1.2")
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/quakeml/1.0")

    def test_xpathWithNamespace(self):
        """
        Tests for XMLParser.xpath
        """
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertEquals(result[0].__module__, 'xml.etree.ElementTree')
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertEquals(result[0].__module__, 'xml.etree.ElementTree')
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertEquals(result[0].__module__, 'xml.etree.ElementTree')
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertEquals(result[0].__module__, 'xml.etree.ElementTree')
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))

    def test_tostring(self):
        """
        Test tostring function.
        """
        # default settings
        # lxml
        el = lxml_etree.Element('test')
        el.append(lxml_etree.Element('test2'))
        result = tostring(el, __etree=lxml_etree)
        self.assertTrue(result.startswith('<?xml'))
        # xml
        el = xml_etree.Element('test')
        el.append(lxml_etree.Element('test2'))
        result = tostring(el, __etree=xml_etree)
        self.assertTrue(result.startswith('<?xml'))
        # 2 - w/o XML declaration
        # lxml
        el = lxml_etree.Element('test')
        el.append(lxml_etree.Element('test2'))
        result = tostring(el, xml_declaration=False, __etree=lxml_etree)
        self.assertTrue(result.startswith('<test>'))
        # xml
        el = xml_etree.Element('test')
        el.append(lxml_etree.Element('test2'))
        result = tostring(el, xml_declaration=False, __etree=xml_etree)
        self.assertTrue(result.startswith('<test>'))


def suite():
    return unittest.makeSuite(XMLWrapperTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
