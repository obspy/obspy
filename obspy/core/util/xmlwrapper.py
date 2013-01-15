# -*- coding: utf-8 -*-
from obspy.core import compatibility
from obspy.core.compatibility import StringIO
import warnings
from lxml import etree
from lxml.etree import register_namespace
import re


def tostring(element, xml_declaration=True, pretty_print=False):
    """
    Generates a string representation of an XML element, including all
    subelements.

    :param element: Element instance.
    :type pretty_print: bool, optional
    :param pretty_print: Enables formatted XML. Defaults to ``False``.
    :return: Encoded string containing the XML data.
    """
    return etree.tostring(element, method="xml", encoding="utf-8",
        pretty_print=pretty_print, xml_declaration=True).decode("utf-8")


class XMLParser:
    """
    Unified wrapper around Python's default xml module and the lxml module.
    """
    def __init__(self, xml_doc, namespace=None):
        """
        Initializes a XMLPaser object.

        :type xml_doc: str, filename, file-like object, parsed XML document
        :param xml_doc: XML document
        :type namespace: str, optional
        :param namespace: Document-wide default namespace. Defaults to ``''``.
        """
        if isinstance(xml_doc, compatibility.string):
            # some string - check if it starts with <?xml
            if xml_doc.strip()[0:5].upper().startswith('<?XML'):
                xml_doc = StringIO(xml_doc)
            # parse XML file
            self.xml_doc = etree.parse(xml_doc)
        elif hasattr(xml_doc, 'seek'):
            # some file-based content
            xml_doc.seek(0)
            self.xml_doc = etree.parse(xml_doc)
        else:
            self.xml_doc = xml_doc
        self.xml_root = self.xml_doc.getroot()
        self.namespace = namespace or self._getRootNamespace()

    def xpath2obj(self, xpath, xml_doc=None, convert_to=str, namespace=None):
        """
        Converts XPath-like query into an object given by convert_to.

        Only the first element will be converted if multiple elements are
        returned from the XPath query.

        :type xpath: str
        :param xpath: XPath string, e.g. ``*/event``.
        :type xml_doc: Element or ElementTree, optional
        :param xml_doc: XML document to query. Defaults to parsed XML document.
        :type convert_to: any type
        :param convert_to: Type to convert to. Defaults to ``str``.
        :type namespace: str, optional
        :param namespace: Namespace used by query. Defaults to document-wide
            namespace set at root.
        """
        try:
            text = self.xpath(xpath, xml_doc, namespace)[0].text
        except IndexError:
            return None
        if text is None:
            return None
        # handle empty nodes
        if text == '':
            return None
        # handle bool extra
        if convert_to == bool:
            if text in ["true", "1"]:
                return True
            elif text in ["false", "0"]:
                return False
            return None
        # try to convert into requested type
        try:
            return convert_to(text)
        except:
            msg = "Could not convert %s to type %s. Returning None."
            warnings.warn(msg % (text, convert_to))
        return None

    def xpath(self, xpath, xml_doc=None, namespace=None):
        """
        Very limited XPath-like query.

        .. note:: This method does not support the full XPath syntax!

        :type xpath: str
        :param xpath: XPath string, e.g. ``*/event``.
        :type xml_doc: Element or ElementTree, optional
        :param xml_doc: XML document to query. Defaults to parsed XML document.
        :type namespace: str, optional
        :param namespace: Namespace used by query. Defaults to document-wide
            namespace set at root.
        :return: List of elements.
        """
        if xml_doc is None:
            xml_doc = self.xml_doc
        if namespace is None:
            namespace = self.namespace
        # namespace handling in lxml as well xml is very limited
        # preserve prefix
        if xpath.startswith('//'):
            prefix = '//'
            xpath = xpath[1:]
        elif xpath.startswith('/'):
            prefix = ''
            xpath = xpath[1:]
        else:
            prefix = ''
        # add namespace to each node
        parts = xpath.split('/')
        xpath = ''
        if namespace:
            for part in parts:
                if part != '*':
                    xpath += "/{%s}%s" % (namespace, part)
                else:
                    xpath += "/%s" % (part)
            xpath = xpath[1:]
        else:
            xpath = '/'.join(parts)
        # restore prefix
        xpath = prefix + xpath
        # lxml
        try:
            return xml_doc.xpath(xpath)
        except:
            pass
        # emulate supports for index selectors (only last element)!
        selector = re.search('(.*)\[(\d+)\]$', xpath)
        if not selector:
            return xml_doc.findall(xpath)
        xpath = selector.groups()[0]
        list_of_elements = xml_doc.findall(xpath)
        try:
            return [list_of_elements[int(selector.groups()[1]) - 1]]
        except IndexError:
            return []

    def _getRootNamespace(self):
        return self._getElementNamespace()

    def _getElementNamespace(self, element=None):
        if element is None:
            element = self.xml_root
        tag = element.tag
        if tag.startswith('{') and '}' in tag:
            return tag[1:].split('}')[0]
        return ''

    def _getFirstChildNamespace(self, element=None):
        if element is None:
            element = self.xml_root
        try:
            element = element.getchildren()[0]
        except:
            return None
        return self._getElementNamespace(element)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
