#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sc3ml events read and write support.

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

from obspy.io.quakeml.core import Pickler, Unpickler, _xml_doc_from_anything
from obspy.io.seiscomp.core import _is_sc3ml as _is_sc3ml_version
from obspy.io.seiscomp.core import validate as validate_sc3ml


SCHEMA_VERSION = ['0.9']


def _is_sc3ml(path_or_file_object):
    """
    Simple function checking if the passed object contains a valid sc3ml file.
    Returns True of False.

    The test is not exhaustive - it only checks the root tag but that should
    be good enough for most real world use cases. If the schema is used to
    test for a StationXML file, many real world files are false negatives as
    they don't adhere to the standard.

    :type path_or_file_object: str
    :param path_or_file_object: File name or file like object.
    :rtype: bool
    :return: ``True`` if SC3ML file is valid.
    """
    return _is_sc3ml_version(path_or_file_object, SCHEMA_VERSION)


def _read_sc3ml(filename, id_prefix='smi:org.gfz-potsdam.de/geofon/'):
    """
    Read a 0.9 SC3ML file and returns a :class:`~obspy.core.event.Catalog`.

    An XSLT file is used to convert the SC3ML file to a QuakeML file. The
    catalog is then generated using the QuakeML module.

    .. warning::
    This function should NOT be called directly, it registers via the
    the :meth:`~obspy.core.event.catalog.Catalog.write` method of an
    ObsPy :class:`~obspy.core.event.catalog.Catalog` object, call this
    instead.

    :type filename: str
    :param filename: SC3ML file to be read.
    :type id_prefix: str
    :param id_prefix: ID prefix. SC3ML does not enforce any particular ID
        restriction, this ID prefix allows to convert the IDs to a well
        formatted QuakeML ID. You can modify the default ID prefix with the
        reverse DNS name of your institute.
    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.

    .. rubric:: Example

    >>> from obspy import read_events
    >>> cat = read_events('/path/to/iris_events.sc3ml')
    >>> print(cat)
    2 Event(s) in Catalog:
    2011-03-11T05:46:24.120000Z | +38.297, +142.373
    2006-09-10T04:26:33.610000Z |  +9.614, +121.961
    """
    xslt_filename = os.path.join(os.path.dirname(__file__), 'data',
                                 'sc3ml_0.9__quakeml_1.2.xsl')
    transform = etree.XSLT(etree.parse(xslt_filename))
    sc3ml_doc = _xml_doc_from_anything(filename)
    quakeml_doc = transform(sc3ml_doc,
                            ID_PREFIX=etree.XSLT.strparam(id_prefix))
    return Unpickler().load(io.BytesIO(quakeml_doc))


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

    if validate and not validate_sc3ml(io.BytesIO(sc3ml_doc), verbose=verbose):
        raise AssertionError("The final SC3ML file did not pass validation.")

    # Open filehandler or use an existing file like object
    try:
        with open(filename, 'wb') as fh:
            fh.write(sc3ml_doc)
    except TypeError:
        filename.write(sc3ml_doc)
