# -*- coding: utf-8 -*-
"""
Directive for creating credits.rst page listing contributors, funds and quotes.
"""

import os


TEMPLATE_CONTRIBUTORS = """.. DON'T EDIT THIS FILE MANUALLY!
   Instead edit CONTRIBUTORS.txt, FUNDS.txt and QUOTES.txt.
   This file will be recreated automatically during building the docs.

Contributors
============

We would like to thank our contributors, whose efforts make this software what
it is. These people have helped by writing code and documentation, and by
testing. They have created and maintained this product, its associated
libraries and applications, our build tools and our web sites.

Hall of Fame
------------

.. hlist::
    :columns: 2

"""

TEMPLATE_FUNDS = """
Funds
-----

ObsPy was partially funded by the

"""

TEMPLATE_QUOTES = """
Quotes
------

"""


# add contributors
def _name_key_function(name):
    """
    Key function for use with :py:func:`sorted` to sort full names given as
    "last name, first names" (e.g. `u'van Driel, Martin\n'`).
    """
    last, first = name.split(",")
    last = last.split()
    # "pop" any last name prefixes that should be ignored during sorting
    if last[0] in ["van"]:
        last.pop(0)
    return last, first


def create_credits_page(app):
    # create credits.rst
    fh = open(os.path.join('source', 'credits.rst'),
              mode='w', encoding='utf-8', newline='\n')
    fh.write(TEMPLATE_CONTRIBUTORS)

    filename = os.path.join(os.pardir, os.pardir, 'obspy', 'CONTRIBUTORS.txt')
    lines = [line for line in open(filename, 'r', encoding='utf-8').readlines()
             if line.strip()]
    contributors = sorted(lines, key=_name_key_function)

    for item in contributors:
        fh.write("    * %s" % (item))
    fh.write(TEMPLATE_FUNDS)

    # add funds
    filename = os.path.join('source', 'credits', 'FUNDS.txt')
    funds = open(filename, 'r', encoding='utf-8').readlines()

    for item in funds:
        fh.write("* %s" % (item))
    fh.write(TEMPLATE_QUOTES)

    # add quotes
    filename = os.path.join('source', 'credits', 'QUOTES.txt')
    funds = open(filename, 'r', encoding='utf-8').readlines()

    for item in funds:
        item = item.split('---')
        # quote
        fh.write("\n\n    %s" % (item[0].strip()))
        # attribution, if given
        try:
            fh.write("\n\n    -- %s" % (item[1]))
        except IndexError:
            pass
    fh.close()


def setup(app):
    app.connect('builder-inited', create_credits_page)
    return {"parallel_write_safe": True,
            "parallel_read_safe": True}
