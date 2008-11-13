# -*- coding: utf-8 -*-


def toAttribute(name):
    """Creates a valid attribute name from a given string."""
    return name.lower().replace(' ','_')


def toXMLTag(name):
    """Creates a XML tag from a given string."""
    temp=name.lower().replace(' ','_')
    temp = temp.replace('fir_', 'FIR_')
    temp = temp.replace('a0_', 'A0_')
    return temp