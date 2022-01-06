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
from obspy.core.util.testing import ImageComparison


# --- ObsPy fixtures


@pytest.fixture(scope='session', autouse=True)
def save_image_directory(request, tmp_path_factory):
    """Create a temporary directory for storing all images."""
    if request.config.getoption('--keep-images'):
        image_save_folder = tmp_path_factory.mktemp('images')
    else:
        image_save_folder = None
    return image_save_folder


@pytest.fixture(scope='function')
def image_comparer(request, save_image_directory):
    """
    Return an image comparison object initiated for specific test.
    """
    def _get_image_name(node_name):
        """Get the expected name of the image."""
        expected_image = node_name.replace('test_', '')
        if expected_image.endswith('_plot'):
            expected_image = expected_image[:-5]
        expected_image += '.png'
        return expected_image

    # get path to modules image folder
    image_dir = Path(request.module.__file__).parent / 'images'
    assert image_dir.is_dir(), f'no image directory found at {image_dir}'
    # get name of node
    expected_image = _get_image_name(request.node.name)
    expected_image_path = image_dir / expected_image
    assert expected_image.exists(), f"no such image {expected_image_path}"
    image_comp = ImageComparison(str(image_dir), expected_image)
    yield image_comp
    # save images if desired
    if save_image_directory:
        pass


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
        "markers", "network: Test requires network resources (internet). "
    )


def pytest_html_report_title(report):
    """Customize the title of the html report (if used) to include version."""
    report.title = f"Obspy Tests ({obspy.__version__})"
