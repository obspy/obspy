# -*- coding: utf-8 -*-
"""
Directive for creating a citations.rst page using BibTeX files in the
bibliography folder.
"""

import glob
import os

from pybtex.database.input import bibtex
from pybtex.style.names.lastfirst import NameStyle
from pybtex.style.template import field, join, node, optional, sentence, words


TEMPLATE = """
.. _citations:

.. DON'T EDIT THIS FILE MANUALLY!
   Instead insert a BibTeX file into the bibliography folder.
   This file will be recreated automatically during building the docs.

Citations
==========

"""


REPLACE_TOKEN = [
    ("<nbsp>", " "),
    ("\xa0", " "),
    ('–', '-'),
    ('—', '-'),
    ('—', '-'),
    ('--', '-'),
    ("\'{a}", "á"),
    ("{\\ae}", "æ"),
    ("**{", "**"),
    ("}**", "**"),
    ("}**", "**"),
]


@node
def names(children, data, role, **kwargs):
    assert not children
    persons = data['entry'].persons[role]
    return join(**kwargs)[[NameStyle().format(person, abbr=True)
                           for person in persons]].format_data(data)


def brackets(data):
    return words(sep='')['(', data, ')']


def bold(data):
    return words(sep='')['**', data, '**']


def italic(data):
    return words(sep='')['*', data, '*']


def format_names(role):
    return words()[names(role, sep=', ', sep2=' and ', last_sep=', and ')]


formats = {
    'article': words(sep='')[
        '\n       | ',
        words(sep=' ')[
            format_names('author'), brackets(field('year'))], ',',
        '\n       | ',
        bold(field('title')), ',',
        '\n       | ',
        sentence(sep=', ')[
            italic(field('journal')),
            optional[words(sep=' ')[
                field('volume'), optional[brackets(field('number'))]]],
            optional[field('pages')],
        ],
        optional['\n       | ', field('url')]
    ],
    'book': words(sep='')[
        '\n       | ',
        words(sep=' ')[
            format_names('author'), brackets(field('year'))], ',',
        '\n       | ',
        bold(field('title')), ',',
        '\n       | ',
        sentence(sep=', ')[
            optional[field('edition')],
            optional[field('series')],
            optional[field('edition')],
            optional[words(sep=' ')[
                'vol.', field('volume'), optional[brackets(field('number'))]]],
            optional[field('pages'), 'pp.'],
            optional[field('publisher')],
            optional[field('address')],
            optional['ISBN: ', field('isbn')],
        ],
        optional['\n       | ', field('url')]
    ],
    'incollection': words(sep='')[
        '\n       | ',
        words(sep=' ')[
            format_names('author'), brackets(field('year'))], ',',
        '\n       | ',
        bold(field('title')), ',',
        '\n       | in ',
        sentence(sep=', ')[
            italic(field('booktitle')),
            optional[field('chapter')],
            optional[words(sep=' ')[
                field('volume'), optional[brackets(field('number'))]]],
            optional[field('pages')],
        ],
        optional['\n       | ', field('url')]
    ],
    'techreport': words(sep='')[
        '\n       | ',
        words(sep=' ')[
            format_names('author'), brackets(field('year'))], ',',
        '\n       | ',
        bold(field('title')), ',',
        '\n       | in ',
        sentence(sep=', ')[
            italic(words(sep=' ')[field('type'), field('number')]),
            field('institution'),
            optional[field('address')],
        ],
        optional['\n       | ', field('url')]
    ],
}


def create_citations_page(app):
    parser = bibtex.Parser(encoding='utf8')
    for file in glob.glob(os.path.join('source', 'bibliography', '*.bib')):
        try:
            parser.parse_file(file)
        except Exception:
            print("Error parsing file %s:" % (file))
            raise

    entries = parser.data.entries

    # create citations.rst
    fh = open(os.path.join('source', 'citations.rst'),
              mode='w', encoding='utf-8', newline='\n')
    fh.write(TEMPLATE)

    for key in sorted(entries.keys()):
        entry = entries[key]
        if entry.type not in formats:
            msg = "BibTeX entry type %s not implemented"
            raise NotImplementedError(msg % (entry.type))
        out = '.. [%s] %s'
        line = str(formats[entry.type].format_data({'entry': entry}))
        # replace special content, e.g. <nbsp>
        for old, new in REPLACE_TOKEN:
            line = line.replace(old, new)
        try:
            fh.write((out % (key, line)))
        except Exception:
            print("Error writing %s:" % (key))
            raise
        fh.write(os.linesep)

    fh.close()


def setup(app):
    app.connect('builder-inited', create_citations_page)
