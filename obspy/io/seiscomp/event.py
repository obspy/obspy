#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sc3ml events write support.

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

import io
import os

from lxml import etree

from obspy.io.quakeml.core import Pickler


def _validate_sc3ml(path_or_object, verbose=False):
    """
    Validates a SC3ML file against the SC3ML 0.9 schema. Returns either True or
    False.

    :param path_or_object: File name or file like object. Can also be an etree
        element.
    :type verbose: bool
    :param verbose: Print error log if True.
    """
    # Get the schema location.
    schema_location = os.path.join(os.path.dirname(__file__), 'data',
                                   'sc3ml_0.9.xsd')
    xmlschema = etree.XMLSchema(etree.parse(schema_location))

    if isinstance(path_or_object, etree._Element):
        xmldoc = path_or_object
    else:
        try:
            xmldoc = etree.parse(path_or_object)
        except etree.XMLSyntaxError:
            if verbose:
                print('Not an XML file')
            return False

    valid = xmlschema.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if verbose and valid is not True:
        print("Error validating SC3ML file:")
        for entry in xmlschema.error_log:
            print("\t%s" % entry)

    return valid


def _write_sc3ml(catalog, filename, validate=False, verbose=False,
                 event_removal=False, **kwargs):  # @UnusedVariable
    """
    Write a SC3ML file. Since a XSLT file is used to write the SC3ML file from
    a QuakeML file, the catalog is first converted in QuakeML.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.catalog.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.catalog.Catalog` object, call this
        instead.

    :type catalog: :class:`~obspy.core.event.catalog.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type validate: bool
    :param validate: If True, the final SC3ML file will be validated against
        the SC3ML schema file. Raises an AssertionError if the validation
        fails.
    :type verbose: bool
    :param verbose: Print validation error log if True.
    :type event_deletion: bool
    :param event_removal: If True, the event elements will be removed. This can
        be useful to associate origins with scevent when injecting SC3ML file
        into seiscomp.
    """
    nsmap_ = getattr(catalog, "nsmap", {})
    quakeml_doc = Pickler(nsmap=nsmap_).dumps(catalog)
    xslt_filename = os.path.join(os.path.dirname(__file__), 'data',
                                 'quakeml_1.2__sc3ml_0.9.xsl')
    transform = etree.XSLT(etree.parse(xslt_filename))
    sc3ml_doc = transform(etree.parse(io.BytesIO(quakeml_doc)))

    # Remove events
    if event_removal:
        for event in sc3ml_doc.xpath("//*[local-name()='event']"):
            event.getparent().remove(event)

    if validate and not _validate_sc3ml(io.BytesIO(sc3ml_doc), verbose):
        raise AssertionError("The final SC3ML file did not pass validation.")

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, "write"):
        file_opened = True
        fh = open(filename, "wb")
    else:
        file_opened = False
        fh = filename

    try:
        fh.write(sc3ml_doc)
    finally:
        # Close if a file has been opened by this function.
        if file_opened is True:
            fh.close()
