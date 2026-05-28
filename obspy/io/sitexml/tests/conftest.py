# -*- coding: utf-8 -*-
"""
SiteXML test fixtures.
"""
import pytest


@pytest.fixture(scope='module')
def testdata(datapath):
    """
    Dictionary with full paths to SiteXML test files by filename.

    SiteXML fixtures are grouped into format subdirectories, so this local
    fixture indexes files recursively while keeping the familiar filename keys.
    """
    files = {
        path.name: path for path in datapath.rglob("*")
        if path.name != ".DS_Store"
    }
    return files
