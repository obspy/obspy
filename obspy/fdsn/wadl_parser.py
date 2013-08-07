from obspy import UTCDateTime

from lxml import etree
import warnings


class WADLParser(object):
    def __init__(self, wadl_string):
        doc = etree.fromstring(wadl_string)
        self._ns = doc.nsmap.values()[0]
        self.parameters = {}

        # Retrieve all the parameters.
        parameters = self._xpath(
            doc, "/application/resources/resource/resource/"
            "method[@id='query'][@name='GET']/request/param")
        if not parameters:
            msg = "Could not find any parameters"
            raise ValueError(msg)

        for param in parameters:
            self.add_parameter(param)

    def add_parameter(self, param_doc):
        name = param_doc.get("name")
        style = param_doc.get("style")
        if style != "query":
            msg = "Unknown parameter style '%s' in WADL" % style
            warnings.warn(msg)

        required = self._convert_boolean(param_doc.get("required"))
        if required is None:
            required = False

        param_type = param_doc.get("type")
        if param_type == "xs:date":
            param_type = UTCDateTime
        elif param_type == "xs:string":
            param_type = str
        elif param_type == "xs:double":
            param_type = float
        elif param_type in ["xs:long", "xs:int"]:
            param_type = int
        # XXX: Remove sboolean once iris fixes it!
        elif param_type in ["xs:boolean", "xs:sboolean"]:
            param_type = bool
        else:
            msg = "Unknown parameter type '%s' in WADL." % param_type
            raise ValueError(msg)

        default_value = self._convert_boolean(param_doc.get("default"))
        if default_value is None:
            default_value = False

        # Parse any possible options.
        options = []
        for option in self._xpath(param_doc, "option"):
            options.append(param_type(option.get("value")))

        doc = ""
        for doc_elem in self._xpath(param_doc, "doc"):
            title = doc_elem.get("title")
            body = doc_elem.text
            doc += "\n%s" % title.strip()
            if body:
                doc += " -- %s" % body.strip()
        doc = doc.strip()

        self.parameters[name] = {
            "required": required,
            "type": param_type,
            "options": options,
            "doc": doc,
            "default_value": default_value}

    @staticmethod
    def _convert_boolean(boolean_string):
        """
        Helper function for boolean value conversion.

        >>> WADLParser._convert_boolean("true")
        True
        >>> WADLParser._convert_boolean("True")
        True

        >>> WADLParser._convert_boolean("false")
        False
        >>> WADLParser._convert_boolean("False")
        False

        >>> WADLParser._convert_boolean("something")
        >>> WADLParser._convert_boolean(1)
        """
        try:
            if boolean_string.lower() == "false":
                return False
            elif boolean_string.lower() == "true":
                return True
            else:
                return None
        except:
            return None

    def _xpath(self, doc, xpath):
        """
        Simple helper method for using xpaths with the default namespace.
        """
        xpath = xpath.replace("/", "/ns:")
        if not xpath.startswith("/ns:"):
            xpath = "ns:" + xpath
        return doc.xpath(xpath, namespaces={"ns": self._ns})


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
