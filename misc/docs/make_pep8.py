# -*- coding: utf-8 -*-

import obspy
from obspy.core.util.testing import check_flake8
import os
from shutil import copyfile


ROOT = os.path.dirname(__file__)
PEP8_IMAGE = os.path.join(ROOT, 'source', 'pep8', 'pep8.png')
PEP8_FAIL_IMAGE = os.path.join(ROOT, 'source', '_static', 'pep8-failing.png')
PEP8_PASS_IMAGE = os.path.join(ROOT, 'source', '_static', 'pep8-passing.png')


path = obspy.__path__[0]

try:
    os.makedirs(os.path.join('source', 'pep8'))
except:
    pass

# write index.rst
fh = open(os.path.join('source', 'pep8', 'index.rst'), 'wt')
fh.write("""
.. _pep8-index:

====
PEP8
====

.. image:: pep8.png

Like most Python projects, we try to adhere to :pep:`8` (Style Guide for Python
Code) and :pep:`257` (Docstring Conventions) with the modifications documented
in the :ref:`coding-style-guide`. Be sure to read those documents if you
intend to contribute code to ObsPy.

Here are the results of the automatic PEP 8 syntax checker:

""")


error_file_count, message, file_count = check_flake8()
message = message.splitlines()
error_classes = set([x.split(" ", 1)[1] for x in message])
statistics = {}
for msg in message:
    err_class = msg.split(" ", 1)[1]
    statistics.setdefault(err_class, 0)
    statistics[err_class] += 1

table_border = "=" * 7 + " " + "=" * max([len(x) for x in error_classes])

if error_file_count == 0:
    fh.write("The PEP 8 checker didn't find any issues.\n")
else:
    fh.write("\n")
    fh.write(".. rubric:: Statistic\n")
    fh.write("\n")
    fh.write(table_border + "\n")
    fh.write("Count   PEP 8 message string                                 \n")
    fh.write(table_border + "\n")
    for err, count in statistics.iteritems():
        fh.write(str(count).ljust(8) + err + "\n")
    fh.write(table_border + "\n")
    fh.write("\n")

    fh.write(".. rubric:: Warnings\n")
    fh.write("\n")
    fh.write("::\n")
    fh.write("\n")

    message = "\n".join(message)
    message = message.replace(path, '    obspy')
    fh.write(message)
    fh.write("\n")

    fh.close()

# remove any old image
try:
    os.remove(PEP8_IMAGE)
except:
    pass
# copy correct pep8 image
if count > 0:
    copyfile(PEP8_FAIL_IMAGE, PEP8_IMAGE)
else:
    copyfile(PEP8_PASS_IMAGE, PEP8_IMAGE)
