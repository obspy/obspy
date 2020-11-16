# -*- coding: utf-8 -*-

import glob
import os

from pybtex.database.input import bibtex
from pybtex.style.names.lastfirst import NameStyle
from pybtex.style.template import field, join, node, optional, sentence, words


REPLACE_TOKEN = [
    (u"<nbsp>", u" "),
    (u"\xa0", u" "),
    (u'–', '-'),
    (u'—', '-'),
    (u'—', '-'),
    (u'--', '-'),
    (u"\'{a}", u"á"),
    (u"{\\ae}", u"æ"),
    (u"**{", u"**"),
    (u"}**", u"**"),
    (u"}**", u"**"),
]


@node
def names(children, data, role, **kwargs):
    assert not children
    persons = data.persons[role]
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
        '\n     - | ',
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
        '\n     - | ',
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
        '\n     - | ',
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
        '\n     - | ',
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

parser = bibtex.Parser(encoding='utf8')

for file in glob.glob(os.path.join('source', 'bibliography', '*.bib')):
    try:
        parser.parse_file(file)
    except Exception:
        print("Error parsing file %s:" % (file))
        raise

entries = parser.data.entries

# write index.rst
fh = open(os.path.join('source', 'citations.rst'), 'wb')
fh.write(b"""
.. _citations:

.. DON'T EDIT THIS FILE MANUALLY!
   Instead insert a BibTeX file into the bibliography folder.
   This file will be recreated automatically during building the docs.
   (The creation of the file can be tested manually by running
    ``make citations`` from command line)

Citations
==========

.. list-table::
   :widths: 1 4

""")

for key in sorted(entries.keys()):
    entry = entries[key]
    if entry.type not in formats:
        msg = "BibTeX entry type %s not implemented"
        raise NotImplementedError(msg % (entry.type))
    out = '   * - .. [%s]%s'
    line = formats[entry.type].format_data(entry).plaintext()
    # replace special content, e.g. <nbsp>
    for old, new in REPLACE_TOKEN:
        line = line.replace(old, new)
    try:
        fh.write((out % (key, line)).encode('UTF-8'))
    except Exception:
        print("Error writing %s:" % (key))
        raise
    fh.write(os.linesep.encode('utf-8'))

fh.close()
