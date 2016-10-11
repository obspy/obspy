# -*- coding: utf-8 -*-

import os
from collections import Counter
from distutils.version import LooseVersion
from shutil import copyfile

import obspy
from obspy.core.util.testing import check_flake8, flake8_version


ROOT = os.path.dirname(__file__)
PEP8_IMAGE = os.path.join(ROOT, 'source', 'pep8', 'pep8.svg')
PEP8_FAIL_IMAGE = os.path.join(ROOT, 'source', '_images', 'pep8-failing.svg')
PEP8_PASS_IMAGE = os.path.join(ROOT, 'source', '_images', 'pep8-passing.svg')


path = obspy.__path__[0]

try:
    os.makedirs(os.path.join('source', 'pep8'))
except:
    pass

report, message = check_flake8()
if flake8_version >= LooseVersion('3.0.0'):
    statistics = Counter()
    for code in 'EWFCN':
        for stat in report.get_statistics(code):
            count, msg = stat.split(maxsplit=1)
            statistics[msg] += int(count)
    statistics = ['%-7d %s' % (count, msg)
                  for msg, count in statistics.most_common()]
    error_count = len(statistics)
else:
    statistics = report.get_statistics()
    error_count = report.get_count()

# write index.rst
head = ("""
.. _pep8-index:

====
PEP8
====

.. image:: pep8.svg

Like most Python projects, we try to adhere to :pep:`8` (Style Guide for Python
Code) and :pep:`257` (Docstring Conventions) with the modifications documented
in the :ref:`coding-style-guide`. Be sure to read those documents if you
intend to contribute code to ObsPy.

Here are the results of the automatic PEP 8 syntax checker:

""")

with open(os.path.join('source', 'pep8', 'index.rst'), 'wt') as fh:
    fh.write(head)

    if error_count == 0:
        fh.write("The PEP 8 checker didn't find any issues.\n")
    else:
        table_border = \
            "=" * 7 + " " + "=" * (max([len(x) for x in statistics]) - 8)
        fh.write("\n")
        fh.write(".. rubric:: Statistic\n")
        fh.write("\n")
        fh.write(table_border + "\n")
        fh.write("Count   PEP 8 message string\n")
        fh.write(table_border + "\n")
        fh.write("\n".join(statistics) + "\n")
        fh.write(table_border + "\n")
        fh.write("\n")

        fh.write(".. rubric:: Warnings\n")
        fh.write("\n")
        fh.write("::\n")
        fh.write("\n")

        message = message.decode().replace(path, '    obspy')
        fh.write(message)
        fh.write("\n")

# remove any old image
try:
    os.remove(PEP8_IMAGE)
except:
    pass
# copy correct pep8 image
if error_count > 0:
    copyfile(PEP8_FAIL_IMAGE, PEP8_IMAGE)
else:
    copyfile(PEP8_PASS_IMAGE, PEP8_IMAGE)
