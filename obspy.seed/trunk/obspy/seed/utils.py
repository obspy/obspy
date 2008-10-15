# -*- coding: utf-8 -*-


def toAttribute(name):
    """Creates a valid attribute name from a given string."""
    return name.lower().replace(' ','_')


def toXMLTag(name):
    """Creates a valid CamelCase XML tag from a given string."""
    return name.title().replace(' ','')