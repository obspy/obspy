# -*- coding: utf-8 -*-
from obspy.core.util.xmlwrapper import XMLParser
import os
import unittest
from lxml import etree as lxml_etree
from xml.etree import ElementTree as xml_etree


class XMLWrapperTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.xmlwrapper
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.iris_xml = os.path.join(path, 'iris_events.xml')
        self.neries_xml = os.path.join(path, 'neries_events.xml')

    def test_getRootNamespace(self):
        """
        Tests for XMLParser._getRootNamespace
        """
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.2")
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.2")
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        self.assertEquals(p._getRootNamespace(),
                          "http://quakeml.org/xmlns/quakeml/1.0")

    def test_getElementNamespace(self):
        """
        Tests for XMLParser._getElementNamespace
        """
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/bed/1.2")
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/bed/1.2")
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        eventParameters = p.xml_root.getchildren()[0]
        self.assertEquals(p._getElementNamespace(eventParameters),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # checking sub elements
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/bed/1.2")
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/quakeml/1.0")
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/bed/1.2")
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        event = p.xml_root.getchildren()[0].getchildren()[0]
        self.assertEquals(p._getElementNamespace(event),
                          "http://quakeml.org/xmlns/quakeml/1.0")

    def test_xpath(self):
        """
        Tests for XMLParser.xpath
        """
        # xml + iris
        xml_doc = xml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertTrue(isinstance(result[0], xml_etree.Element))
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertTrue(isinstance(result[0], xml_etree.Element))
        # lxml + iris
        xml_doc = lxml_etree.parse(self.iris_xml)
        p = XMLParser(xml_doc=xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 2)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))
        # xml + neries
        xml_doc = xml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertTrue(isinstance(result[0], xml_etree.Element))
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertTrue(isinstance(result[0], xml_etree.Element))
        # lxml + neries
        xml_doc = lxml_etree.parse(self.neries_xml)
        p = XMLParser(xml_doc=xml_doc)
        ns = p._getFirstChildNamespace()
        result = p.xpath('*/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))
        result = p.xpath('eventParameters/event', namespace=ns)
        self.assertEquals(len(result), 3)
        self.assertTrue(isinstance(result[0], lxml_etree._Element))


def suite():
    return unittest.makeSuite(XMLWrapperTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
