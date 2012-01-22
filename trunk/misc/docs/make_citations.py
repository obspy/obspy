# -*- coding: utf-8 -*-

from pybtex.database import BibliographyData
from pybtex.database.input import bibtex
from pybtex.style.formatting.unsrt import Style
from pybtex.style.names.lastfirst import NameStyle
from pybtex.style.template import sentence, field, optional, words, node, join
import glob
import os

# write index.rst
fh = open(os.path.join('source', 'citations.rst'), 'wt')
fh.write("""
.. _citations:

Citations
==========

""")

parser = bibtex.Parser(encoding='utf8')
for file in glob.glob(os.path.join('source', 'bibliography', '*.bib')):
    parser.parse_file(file)

@node
def names(children, data, role, **kwargs):
    assert not children
    persons = data.persons[role]
    return join(**kwargs) [[NameStyle().format(person, abbr=True)
                            for person in persons]].format_data(data)

def brackets(data):
    return words(sep='')['(', data, ')']


def bold(data):
    return words(sep='')['**', data, '**']


def italic(data):
    return words(sep='')['*', data, '*']


def format_names(role):
    return words()[names(role, sep=', ', sep2 = ' and ', last_sep=', and ')]


formats = {
    'article': words(sep='') [
        '\n   | ',
        words(sep=' ')[
            format_names('author'), brackets(field('year'))], ',',
        '\n   | ',
        bold(field('title')), ',',
        '\n   | ',
        sentence(sep=', ')[
            italic(field('journal')),
            optional [words(sep=' ')[
                field('volume'), optional [brackets(field('number'))]]],
            optional [field('pages')],
        ],
        optional ['\n   | ', field('url')]
    ],
    'book': words(sep='') [
        '\n   | ',
        words(sep=' ')[
            format_names('author'), brackets(field('year'))], ',',
        '\n   | ',
        bold(field('title')), ',',
        '\n   | ',
        sentence(sep=', ')[
            optional [field('edition')],
            optional [field('series')],
            optional [field('edition')],
            optional [words(sep=' ')[
                'vol.', field('volume'), optional [brackets(field('number'))]]],
            optional [field('pages'), 'pp.'],
            optional [field('publisher')],
            optional [field('address')],
            optional ['ISBN: ', field('isbn')],
        ],
        optional ['\n   | ', field('url')]
    ]
}


for key, entry in parser.data.entries.iteritems():
    if entry.type not in formats:
        msg = "BibTeX entry type %s not implemented"
        raise NotImplementedError(msg % (entry.type))
    fh.write('.. [%s] ' % (key))
    line = formats[entry.type].format_data(entry).plaintext()
    # replace special content, e.g. <nbsp>
    line = line.replace('<nbsp>', ' ')
    fh.write(line)
    fh.write(os.linesep)

fh.close()
