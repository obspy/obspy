# -*- coding: utf-8 -*-
#
# ObsPy documentation build configuration file, created by
# sphinx-quickstart on Sat Dec 26 11:55:06 2009.
# :note: major modification of original version at the end
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import sys, os

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#sys.path.append(os.path.abspath(os.path.dirname(__file__)+'../../../'))

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary', 'numpydoc',
              'matplotlib.sphinxext.plot_directive',
              'matplotlib.sphinxext.only_directives']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General substitutions.
project = 'ObsPy'
copyright = '2009-2011, ObsPy Development Team'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
version = '0.2'
# The full version, including alpha/beta/rc tags.
release = '0.2'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%B %d %H o'clock, %Y"

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
#exclude_dirs = []

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'default.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "ObsPy documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%Y-%m-%dT%H:%M:%S'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'ObsPydoc'


# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
  ('index', 'ObsPy.tex', 'ObsPy Documentation',
   'ObsPy Development Team', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
#latex_preamble = ''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

# Redefine how to process inherited methods/members
#def process_inherited(app, what, name, obj, options, docstringlines):
#    """ 
#    If we're including inherited members, omit their docstrings. 
#    """
#    if not options.get('inherited-members'):
#        return
#
#    if what in ['class', 'data', 'exception', 'function', 'module']:
#        return
#
#    name = name.split('.')[-1]
#
#    if what == 'method' and hasattr(obj, 'im_class'):
#        if name in obj.im_class.__dict__.keys():
#            return
#
#    if what == 'attribute' and hasattr(obj, '__objclass__'):
#        if name in obj.__objclass__.__dict__.keys():
#            return
#
#    for i in xrange(len(docstringlines)):
#        docstringlines.pop()

# Options for Including private Members/Methods
#----------------------------------------------
# For reference, see
# * http://bitbucket.org/birkenfeld/sphinx/src/tip/tests/test_autodoc.py
# * http://hg.sagemath.org/sage-main/file/21efb0b3fc47/doc/common/conf.py#l1
#   which is the sagemath conf.py
# * http://trac.sagemath.org/sage_trac/attachment/ticket/7549/\
#       trac_7549-doc_inheritance_underscore_v3.patch

# Do not skip private members
#def skip_underscore(app, what, name, obj, skip, options): 
#    """ 
#    Conditionally include docstrings for objects whose names begin 
#    with one underscore ('_'). 
#    """ 
#    name = name.split('.')[-1]
#    if name.startswith('_') and not name.startswith('__'):
#        return False 
#    return skip 

autosummary_generate = ["obspy.events.rst"]
exclude_trees = ['_templates']

from numpydoc.docscrape_sphinx import SphinxDocString
from numpydoc.docscrape import Reader, NumpyDocString
import textwrap

# Monkey path NumpyDocString
def obspy__init__(self, docstring, config={}):
    docstring = textwrap.dedent(docstring).split('\n')
    newdoc = []
    params = {}
    ids = []
    kw = None
    for line in docstring:
        if not line or line.isspace():
            kw = None
            newdoc.append(line)
            continue
        elif not kw and line[0] not in [':', '@']:
            kw = None
            newdoc.append(line)
            continue
        if line.startswith(":param ") or line.startswith("@param "):
            kw, desc = line[7:].split(':', 1)
            kw = kw.strip()
            ids.append(kw)
            params.setdefault(kw, {})
            params[kw].setdefault('dtype', '')
            params[kw]['desc'] = ['    ' + desc.strip()]
            continue
        elif line.startswith(":type ") or line.startswith("@type "):
            kw, dtype = line[6:].split(':', 1)
            kw = kw.strip()
            params.setdefault(kw, {})
            params[kw].setdefault('desc', [])
            params[kw]['dtype'] = dtype
            continue
        elif line.startswith(":return:") or line.startswith("@return:"):
            desc = line[8:].strip()
            kw = 'return'
            params.setdefault(kw, {})
            params[kw].setdefault('dtype', '')
            params[kw]['desc'] = ['    ' + desc.strip()]
            continue
        elif line.startswith(":rtype:") or line.startswith("@rtype:"):
            dtype = line[7:].strip()
            kw = 'return'
            params.setdefault(kw, {})
            params[kw].setdefault('desc', [])
            params[kw]['dtype'] = dtype
            continue
        if not kw:
            newdoc.append(line)
            continue
        params[kw]['desc'].append('    ' + line.strip())

    docstring = newdoc
    if ids:
        docstring += ['Parameters', '----------', '']
    for id in ids:
        docstring += ["%s : %s" % (id, params[id]['dtype'])]
        docstring += params[id]['desc']
    if 'return' in params:
        docstring += ['', 'Returns', '-------', '']
        docstring += ["%s" % params['return']['dtype']]
        docstring += params['return']['desc']
    docstring.append('')


    self._doc = Reader(docstring)
    self._parsed_data = {
        'Signature': '',
        'Basic Usage': [],
        'Summary': [''],
        'Extended Summary': [],
        'Parameters': [],
        'Returns': [],
        'Raises': [],
        'Warns': [],
        'Other Parameters': [],
        'Attributes': [],
        'Methods': [],
        'See Also': [],
        'Notes': [],
        'Supported Operations': [],
        'Warnings': [],
        'References': '',
        'Examples': '',
        'Example': '',
        'index': {}
        }

    self._parse()
    self['Supported Operations'] = self._parse_param_list(self['Supported Operations'])

def obspy_str_param_list(self, name):
    out = []
    if self[name]:
        out += self._str_field_list(name)
        out += ['']
        for param, param_type, desc in self[name]:
            if not param_type:
                out += self._str_indent(['%s' % (param.strip())])
            else:
                out += self._str_indent(['**%s** : %s' % (param.strip(),
                                                          param_type.strip())])
            out += ['']
            out += self._str_indent(desc, 8)
            out += ['']
    return out

def obspy_str_table(self, name, h1, h2):
    out = []
    if self[name]:
        out += self._str_header(name)
        out += ['']
        # table header
        max1 = max([len(p[0]) for p in self[name]])
        out += ['=' * max1 + ' ' + '=' * 80]
        out += [h1 + ' ' * (max1 + 1 - len(h1)) + h2]
        out += ['=' * max1 + ' ' + '=' * 80]
        for param, param_type, desc in self[name]:
            d1 = param.strip()
            d1 = d1 + ' ' * (max1 + 1 - len(d1))
            out += [d1 + desc[0].strip()]
            if len(desc) == 1:
                continue
            for d2 in desc[1:]:
                out += [' ' * max1 + ' ' + d2.strip()]
        out += ['=' * max1 + ' ' + '=' * 80]
        out += ['']
    return out

def obspy_str_examples(self, name):
    examples_str = "\n".join(self[name])
    if (self.use_plots and 'import matplotlib' in examples_str
            and 'plot::' not in examples_str):
        out = []
        out += self._str_header(name)
        out += ['.. plot::', '']
        out += self._str_indent(self[name])
        out += ['']
        return out
    else:
        return self._str_section(name)

def obspy__str__(self, indent=0, func_role="obj"):
    out = []
    out += self._str_signature()
    out += self._str_index() + ['']
    out += self._str_summary()
    out += self._str_extended_summary()
    for param_list in ('Parameters', 'Returns', 'Raises'):
        out += self._str_param_list(param_list)
    out += self._str_examples('Basic Usage')
    out += self._str_member_list('Attributes')
    out += self._str_warnings()
    out += self._str_see_also(func_role)
    out += self._str_table('Supported Operations', 'Operation', 'Result')
    out += self._str_examples('Notes')
    out += self._str_references()
    out += self._str_examples('Examples')
    out += self._str_examples('Example')
    # Do not include Methods, these are generated by
    # sphinx-autogen automatically. If included here,
    # the will be included twice
    #out += self._str_member_list('Methods')
    out = self._str_indent(out, indent)
    return '\n'.join(out)

NumpyDocString.__init__ = obspy__init__
SphinxDocString.__str__ = obspy__str__
SphinxDocString._str_param_list = obspy_str_param_list
SphinxDocString._str_examples = obspy_str_examples
SphinxDocString._str_table = obspy_str_table

#from sphinx.ext.autodoc import cut_lines
# Attach this to the builder
def setup(app):
    # numpydoc.py (0.4) adds a extra NumpyPythonDomain domain which causes
    # an ``TypeError: adding ViewList to a non-ViewList`` for ``.. plot::``
    # directives inside the docstrings of functions. Deactivate it
    # NOTE: this is the second time numpydoc causes troubles, if there are
    # any problems try the version shipped with matplotlibs sampledoc_tut
    # as reference:
    # https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/\
    # trunk/sampledoc_tut/sphinxext/numpydoc.py
    try:
        app.domains.pop('np')
    except:
        pass
    #app.connect('autodoc-process-docstring', cut_lines(2))
    #app.connect('autodoc-process-signature', mangle_signature)
    #app.connect('autodoc-process-docstring', process_inherited)
    #app.connect('autodoc-skip-member', skip_underscore)
