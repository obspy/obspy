#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SC3ML function used for both inventory and event module.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import re
import warnings

from lxml import etree

from obspy.io.quakeml.core import _xml_doc_from_anything


# SC3ML version for which an XSD file is available
SUPPORTED_XSD_VERSION = ['0.7', '0.8', '0.9']


def _is_sc3ml(path_or_file_object, schema_versions):
    """
    Simple function checking if the passed object contains a valid sc3ml file
    according to the list of versions given in parameters. Returns True of
    False.

    The test is not exhaustive - it only checks the root tag but that should
    be good enough for most real world use cases. If the schema is used to
    test for a StationXML file, many real world files are false negatives as
    they don't adhere to the standard.

    :type path_or_file_object: str
    :param path_or_file_object: File name or file like object.
    :type schema_versions: list of strings
    :param schema_versions: List of supported SC3ML version.
    :rtype: bool
    :return: `True` if file is a SC3ML file.
    """
    if hasattr(path_or_file_object, "tell") and hasattr(path_or_file_object,
                                                        "seek"):
        current_position = path_or_file_object.tell()

    if isinstance(path_or_file_object, etree._Element):
        xmldoc = path_or_file_object
    else:
        try:
            xmldoc = _xml_doc_from_anything(path_or_file_object)
        except ValueError:
            return False
        finally:
            # Make sure to reset file pointer position.
            try:
                path_or_file_object.seek(current_position, 0)
            except Exception:
                pass

    if hasattr(xmldoc, "getroot"):
        root = xmldoc.getroot()
    else:
        root = xmldoc

    match = re.match(
        r'{http://geofon\.gfz-potsdam\.de/ns/seiscomp3-schema/([-+]?'
        r'[0-9]*\.?[0-9]+)}', root.tag)

    if match is None:
        return False

    # Check if schema version is supported
    version = match.group(1)
    if version not in schema_versions:
        warnings.warn('The sc3ml file has version %s, ObsPy can '
                      'deal with versions [%s]. Proceed with caution.' % (
                          version, ', '.join(schema_versions)))

    return True


def validate(path_or_object, version='0.9', verbose=False):
    """
    Check if the given file is a valid SC3ML file.

    :type path_or_object: str
    :param path_or_object: File name or file like object. Can also be an etree
        element.
    :type version: str
    :param version: Version of the SC3ML schema to validate against.
    :type verbose: bool
    :param verbose: Print error log if True.
    :rtype: bool
    :return: `True` if SC3ML file is valid.
    """
    if version not in SUPPORTED_XSD_VERSION:
        raise ValueError('%s is not a supported version. Use one of these '
                         'versions: %s' % (version, SUPPORTED_XSD_VERSION))

    # Get the schema location.
    xsd_filename = 'sc3ml_%s.xsd' % version
    schema_location = os.path.join(os.path.dirname(__file__), 'data',
                                   xsd_filename)
    xmlschema = etree.XMLSchema(etree.parse(schema_location))

    if hasattr(path_or_object, "tell") and hasattr(path_or_object, "seek"):
        current_position = path_or_object.tell()

    if isinstance(path_or_object, etree._Element):
        xmldoc = path_or_object
    else:
        try:
            xmldoc = _xml_doc_from_anything(path_or_object)
        except ValueError:
            return False
        finally:
            # Make sure to reset file pointer position.
            try:
                path_or_object.seek(current_position, 0)
            except Exception:
                pass

    valid = xmlschema.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if verbose and valid is not True:
        print("Error validating SC3ML file:")
        for entry in xmlschema.error_log:
            print("\t%s" % entry)

    return valid
