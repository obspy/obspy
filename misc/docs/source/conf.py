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

import cartopy  # NOQA  Do we really need this?

import matplotlib

# Use matplotlib agg backend
matplotlib.use("agg")


# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.append(os.path.abspath('_ext'))


# -- Project information -----------------------------------------------------
project = 'ObsPy'
author = 'The ObsPy Development Team (devs@obspy.org)'
year = datetime.date.today().year
copyright = '2012-{}, The ObsPy Development Team (devs@obspy.org)'.format(year)
version = ".".join(obspy.__version__.split(".")[:3])
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
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.duration',
    'sphinx.ext.todo',
    # theme
    'm2r2',
    'sphinx_rtd_theme',
    # custom extensions
    'plot_directive',
    'credits',
    'citations',
    # 'sphinxcontrib.bibtex',
]

# Uncomment this if you want to use sphinxcontrib.bibtex
# from pathlib import Path
# bibtex_bibfiles = (
#     list(Path('.').joinpath('source', 'bibliography').glob('*.bib'))
# )

# The file extensions of source files. Sphinx considers the files with this
# suffix as sources. The value can be a dictionary mapping file extensions to
# file types.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_ext', '_images', '_static', '_templates', 'bibliography', 'credits']

# Warn about all references where the target cannot be found.
nitpicky = True
nitpick_ignore = [
    ('py:class', 'optional'),
    ('py:class', 'file'),
    ('py:class', 'file-like object'),
    ('py:class', 'open file'),
    ('py:class', 'valid matplotlib color'),
    ('py:class', 'valid matplotlib colormap'),
    ('py:class', 'same class as original object'),
    ('py:class', 'sqlalchemy.orm.decl_api.Base'),
    ('py:class', 'array_like'),
    ('py:class', 'hashable'),
    ('py:class', 'shapefile.Writer'),
    ('py:class', 'collections.namedtuple')
]

# suppress built-in types by default in nitpick
import builtins
for name in dir(builtins):
    nitpick_ignore += [('py:class', name)]

# suppress warnings for all Class built using the _event_type_class_factory
# which generate: "py:class reference target not found:
# obspy.core.event.base._event_type_class_factory.<locals>.
# AbstractEventTypeWithResourceID
nitpick_ignore_regex = [
    (r'py:class', r'.*<locals>\.AbstractEvent..*'),
    (r'py:class', r'math\..*'),
]

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sqlalchemy': ('https://docs.sqlalchemy.org/en/latest/', None),
    'pip': ('https://pip.pypa.io/en/stable/', None),
    'lxml': ('https://lxml.de/apidoc/', None),
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'sqlalchemy': ('https://docs.sqlalchemy.org/en/latest/', None),
    'requests': ('https://docs.python-requests.org/en/latest/', None)
}

# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined).
add_module_names = False

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['obspy.']

# These values determine how to format the current date.
today_fmt = "%B %d %H o'clock, %Y"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

html_logo = '_static/obspy_logo_no_text.svg'
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.min.css',
]

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
}

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%Y-%m-%dT%H:%M:%S'


# If true, a list all whose items consist of a single paragraph and/or a
# sub-list all whose items etc… (recursive definition) will not use the <p>
# element for any of its items.
html_compact_lists = True


# -- Options for LaTeX output -------------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual])
latex_documents = [
  ('tutorial/index', 'ObsPyTutorial.tex', u'ObsPy Tutorial',
   u'The ObsPy Development Team (devs@obspy.org)', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Additional stuff for the LaTeX preamble.
# latex_preamble = ''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'obspydocumentation', u'ObsPy Documentation',
     [u'The ObsPy Development Team (devs@obspy.org)'], 1)
]


# -- Options for autodoc / autosummary exensions -----------------------------

# Don't merge __init__ method in auoclass content
autoclass_content = 'class'

# generate automatically stubs
autosummary_generate = True

# If true, autosummary already overwrites stub files by generated contents.
autosummary_generate_overwrite = False

# Not sure this will remove warnings from collections's Mapping (Attribdict)
autodoc_inherit_docstrings = False

# -- Options for linkcheck exension ------------------------------------------
linkcheck_timeout = 5
linkcheck_workers = 10


# -- Options for matplotlib plot directive -----------------------------------
# File formats to generate.
plot_formats = [('png', 110), ]  # ('hires.png', 200), ('pdf', 200)]
