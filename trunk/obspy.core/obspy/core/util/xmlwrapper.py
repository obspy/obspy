# -*- coding: utf-8 -*-
import StringIO
import warnings
try:
    # try using lxml as it is faster
    from lxml import etree
except ImportError:
    from xml.etree import ElementTree as etree  # @UnusedImport
import re


RE_ENDS_WITH_SELECTOR = re.compile('^.*/[/d/]$')


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
        if isinstance(xml_doc, basestring):
            # some string - check if it starts with <?xml
            if xml_doc.strip()[0:5].upper().startswith('<?XML'):
                xml_doc = StringIO.StringIO(xml_doc)
            # parse XML
            self.xml_doc = etree.parse(xml_doc)
        elif hasattr(xml_doc, 'seek'):
            self.xml_doc = etree.parse(xml_doc)
        else:
            self.xml_doc = xml_doc
        self.xml_root = self.xml_doc.getroot()
        self.namespace = namespace or self._getRootNamespace()

    def xml2obj(self, xpath, xml_doc=None, convert_to=str, namespace=None):
        """
        XPath query.

        :type xpath: str
        :param xpath: XPath string, e.g. ``*/event``.
        :type xml_doc: Element or ElementTree, optional
        :param xml_doc: XML document to query. Defaults to parsed XML document.
        :type namespace: str, optional
        :param namespace: Namespace used by query. Defaults to document-wide
            namespace set at root.
        """
        try:
            text = self.xpath(xpath, xml_doc, namespace)[0].text
        except IndexError:
            # str(None) should be ''
            if convert_to == str:
                return ''
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
        for part in parts:
            xpath += '/'
            if part == '':
                part = '*'
            if part != '*':
                xpath += '%(ns)s'
            xpath += part
        # restore prefix
        xpath = prefix + xpath[1:]
        # lxml
        try:
            return xml_doc.xpath(xpath % ({'ns': 'ns:'}), {'ns': namespace})
        except:
            pass
        if namespace:
            xpath = xpath % ({'ns': '{' + namespace + '}'})
        else:
            xpath = xpath % ({'ns': ''})
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
