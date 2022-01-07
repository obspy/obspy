#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Obspy's testing configuration file.
"""
from pathlib import Path

import numpy as np
import pytest

import obspy
from obspy.core.util import NETWORK_MODULES


# --- ObsPy fixtures

@pytest.fixture(scope='class')
def ignore_numpy_errors():
    """ Ignore numpy errors for marked functions/classes. """
    nperr = np.geterr()
    np.seterr(all='ignore')
    yield
    np.seterr(**nperr)


@pytest.fixture(scope='session', autouse=True)
def save_image_directory(request, tmp_path_factory):
    """Create a temporary directory for storing all images."""
    tmp_image_path = tmp_path_factory.mktemp('images')
    yield Path(tmp_image_path)
    # if keep images is selected then we move images to directory
    # and add info about environment for upload.
    # breakpoint()


@pytest.fixture(scope='function')
def image_path(request, save_image_directory):
    """
    Return an image comparison object initiated for specific test.
    """
    parent_name = getattr(request.node.parent, 'name', '')
    node_name = request.node.name
    if parent_name:  # add parent class to name
        node_name = parent_name + '__' + node_name
    new_path = save_image_directory / (node_name + '.png')
    return new_path


# --- Pytest configuration


def pytest_addoption(parser):
    """Pytest hook which allows setting package-specfic command-line args."""
    parser.addoption('--network', action='store_true', default=False,
                     help='test network modules', )
    parser.addoption('--all', action='store_true', default=False,
                     help='run both network and non-network tests', )
    parser.addoption('--coverage', action='store_true', default=False,
                     help='Report Obspy Coverage and generate xml report', )
    parser.addoption('--report', action='store_true', default=False,
                     help='Generate and HTML report of test results', )
    parser.addoption('--keep-images', action='store_true',
                     help='store images created during image comparison '
                          'tests in subfolders of baseline images')


def pytest_collection_modifyitems(config, items):
    """ Preprocessor for collected tests. """
    network_nodes = set(NETWORK_MODULES)

    for item in items:
        # get the obspy model test originates from (eg clients.arclink)
        obspy_node = '.'.join(item.nodeid.split('/')[1:3])
        # if test is a network test apply network marker
        # We need to keep these to properly mark doctests, event though
        # the test files now have proper marks.
        if obspy_node in network_nodes:
            item.add_marker(pytest.mark.network)


def pytest_runtest_setup(item):
    """Setup test runs."""
    # skip network tests if not specified by command line argument.
    if 'network' in item.keywords and not item.config.getvalue("--network"):
        pytest.skip("need --network option to run")


def pytest_configure(config):
    """
    Configure pytest with custom logic for ObsPy before test run.
    """
    # add doctest option
    config.option.doctestmodules = True

    # skip or select network options based on options
    network_selected = config.getoption('--network')
    all_selected = config.getoption('--all')
    if network_selected and not all_selected:
        setattr(config.option, 'markexpr', 'network')
    elif not network_selected and not all_selected:
        setattr(config.option, 'markexpr', 'not network')

    # select appropriate options for report
    # this is the same as --html=obspy_report.html --self-contained-html
    if config.getoption('--report'):
        config.option.htmlpath = 'obspy_report.html'
        config.option.self_contained_html = True

    # select options for coverage
    # this is the same as using:
    # --cov obspy --cov-report term-missing --cov-report='xml' --cov-append
    if config.getoption('--coverage'):
        config.option.cov_report = {'term-missing': None, 'xml': None}
        config.known_args_namespace.cov_source = ['obspy']
        config.option.cov_append = True
        # this is a bit hinky, but we need to register and load coverage here
        # or else it doesn't work
        from pytest_cov.plugin import CovPlugin
        config.option.cov_source = ['obspy']
        options = config.known_args_namespace
        plugin = CovPlugin(options, config.pluginmanager)
        config.pluginmanager.register(plugin, '_cov')

    # set print options
    try:
        np.set_printoptions(legacy='1.13')
    except (TypeError, AttributeError):
        pass
    # ensure matplotlib doesn't print anything to the screen
    import matplotlib
    matplotlib.use('Agg')

    # add marker, having pytest.ini messed with configuration
    config.addinivalue_line(
        "markers", "network: Test requires network resources (internet)."
    )
    config.addinivalue_line(
        "markers", "image: Test produces a matplotlib image."
    )


def pytest_itemcollected(item):
    """ we just collected a test item. """
    # automatically apply image mark if image_path is used
    if 'image_path' in getattr(item, 'fixturenames', {}):
        item.add_marker('image')


def pytest_html_report_title(report):
    """Customize the title of the html report (if used) to include version."""
    report.title = f"Obspy Tests ({obspy.__version__})"
