#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seiscomp.core test suite.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import re

import pytest

from obspy.io.seiscomp.core import _is_scml, validate


class TestCore():
    """
    Test suite for obspy.io.seiscomp.event
    """
    def test_scml_versions(self, testdata):
        """
        Test multiple schema versions
        """
        for version in ['0.10', '0.11', '0.12', '0.13', '0.14']:
            filename = testdata['version%s' % version]
            assert _is_scml(filename)

    def test_scml_no_version_attribute(self, testdata):
        filename = testdata['no_version_attribute.sc3ml']
        assert _is_scml(filename)

    def test_validate(self, testdata):
        filename = testdata['qml-example-1.2-RC3.sc3ml']
        assert validate(filename)
        assert not validate(filename, version='0.8')

        expected_error = re.escape(
            "0.99 is not a supported version. Use one of these"
            " versions: [0.7, 0.8, 0.9, 0.10, "
            "0.11, 0.12, 0.13, 0.14].")
        with pytest.raises(ValueError, match=expected_error):
            validate(filename, version='0.99')
