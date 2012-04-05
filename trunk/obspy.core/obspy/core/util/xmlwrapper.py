# -*- coding: utf-8 -*-
import warnings
try:
    # try using lxml as it is faster
    from lxml import etree
except ImportError:
    from xml.etree import ElementTree as etree  # @UnusedImport


class XMLParser:
    """
    Unified wrapper around Python's default xml module and the lxml module.
    """
    def __init__(self, filename=None, xml_doc=None, namespace=None):
        if filename:
            self.xml_doc = etree.parse(filename)
        elif xml_doc:
            self.xml_doc = xml_doc
        self.xml_root = self.xml_doc.getroot()
        self.namespace = str(namespace) or self._getRootNamespace()

    def xml2obj(self, xpath, xml_doc=None, convert_to=None, namespace=None):
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
        if convert_to is None:
            result = self.xpath(xpath, xml_doc=xml_doc, namespace=namespace)
            return result
        if xml_doc is None:
            xml_doc = self.xml_doc
        if namespace is None:
            namespace = self.namespace
        parts = xpath.split('/')
        ns = '/{%s}' % (namespace)
        xpath = (ns + ns.join(parts))[1:]
        text = xml_doc.findtext(xpath)
        # str(None) should be ''
        if convert_to == str and text is None:
            return ''
        if text is None:
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
        XPath query.

        :type xpath: str
        :param xpath: XPath string, e.g. ``*/event``.
        :type xml_doc: Element or ElementTree, optional
        :param xml_doc: XML document to query. Defaults to parsed XML document.
        :type namespace: str, optional
        :param namespace: Namespace used by query. Defaults to document-wide
            namespace set at root.
        """
        if xml_doc is None:
            xml_doc = self.xml_doc
        if namespace is None:
            namespace = self.namespace
        # delete leading slash
        if xpath.startswith('/'):
            xpath = xpath[1:]
        # build up query
        parts = xpath.split('/')
        xpath = ''
        for part in parts:
            xpath += '/'
            if part != '*':
                xpath += '%(ns)s'
            xpath += part
        xpath = xpath[1:]
        # lxml
        try:
            return xml_doc.xpath(xpath % ({'ns': 'ns:'}), {'ns': namespace})
        except:
            pass
        # xml
        return xml_doc.findall(xpath % ({'ns': '{' + namespace + '}'}))

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
