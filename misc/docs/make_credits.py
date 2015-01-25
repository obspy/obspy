# -*- coding: utf-8 -*-

import codecs
import os


# generate credits
fh = codecs.open(os.path.join('source', 'credits.rst'), 'w', 'utf-8')
fh.write(""".. DON'T EDIT THIS FILE MANUALLY!
   Instead edit txt files in the credits folder and
   run ``make credits`` from command line to automatically create this file!

Contributors
============

We would like to thank our contributors, whose efforts make this software what
it is. These people have helped by writing code and documentation, and by
testing. They have created and maintained this product, its associated
libraries and applications, our build tools and our web sites.

.. rubric:: Hall of Fame

.. hlist::
    :columns: 3

""")

# add contributors
filename = os.path.join(os.pardir, os.pardir, 'CONTRIBUTORS.txt')
contributors = sorted(codecs.open(filename, 'r', 'utf-8').readlines())

for item in contributors:
    fh.write("    * %s" % (item))

fh.write("""
.. rubric:: Funds

ObsPy was partially funded by the

""")

# add funds
filename = os.path.join('source', 'credits', 'FUNDS.txt')
funds = codecs.open(filename, 'r', 'utf-8').readlines()

for item in funds:
    fh.write("* %s" % (item))

fh.write("""
.. rubric:: Quotes

""")

# add quotes
filename = os.path.join('source', 'credits', 'QUOTES.txt')
funds = codecs.open(filename, 'r', 'utf-8').readlines()

for item in funds:
    item = item.split('---')
    fh.write("""
.. epigraph::
    %s""" % (item[0]))
    try:
        fh.write("""

    -- %s""" % (item[1]))
    except IndexError:
        pass

fh.close()
