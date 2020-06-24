# -*- coding: utf-8 -*-
#
# ObsPy documentation configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#

import datetime
import obspy
import os
import sys

import matplotlib

import sphinx_rtd_theme


# Use matplotlib agg backend
matplotlib.use("agg")


# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath('.') + os.sep + '_ext')


# -- Project information -----------------------------------------------------
project = 'ObsPy'
author = 'The ObsPy Development Team (devs@obspy.org)'
year = datetime.date.today().year
copyright = '%d, The ObsPy Development Team (devs@obspy.org)' % (year)
version = ".".join(obspy.__version__.split(".")[:2])
release = obspy.__version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.1.1'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.coverage',
    'sphinx.ext.duration',
    'sphinx.ext.todo',
    # theme
    'sphinx_rtd_theme',
    # custom extensions
    'plot_directive',
    'credits',
    'citations',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Warn about all references where the target cannot be found.
nitpicky = True
nitpick_ignore = ['norm_resp']

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sqlalchemy': ('https://docs.sqlalchemy.org/en/latest/', None),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_logo = '_static/images/obspy_logo_no_text.svg'
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path 
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]
