# -*- coding: utf-8 -*-

from pep8 import input_file, input_dir, get_statistics, get_count, \
    process_options
import obspy
import os
import sys
from StringIO import StringIO

paths = [p for p in obspy.__path__ if 'fissures' not in p]
modules = [p.split(os.sep)[-2] for p in paths]


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

Like most Python projects, we try to adhere to :pep:`8` (Style Guide for Python
Code) and :pep:`257` (Docstring Conventions) with the modifications documented
in the :ref:`coding-style-guide`. Be sure to read those documents if you
intend to contribute code to ObsPy.

Here are the results of the automatic PEP 8 syntax checker:

.. toctree::

""")
for module in sorted(modules):
    fh.write('   %s\n' % module)
fh.close()

# backup stdout
stdout = sys.stdout

# handle each module
for path in paths:
    module = path.split(os.sep)[-2]
    sys.stdout = StringIO()
    # clean up runner options
    options, args = process_options()
    options.repeat = True
    if os.path.isdir(path):
        input_dir(path, runner=input_file)
    elif not excluded(path):
        input_file(path)
    sys.stdout.seek(0)
    data = sys.stdout.read()
    statistic = get_statistics('')
    count = get_count()
    # write rst file
    fh = open(os.path.join('source', 'pep8', module + '.rst'), 'wt')
    title = "%s (%d)" % (module, count)
    fh.write(len(title) * "=" + "\n")
    fh.write(title + "\n")
    fh.write(len(title) * "=" + "\n")
    fh.write("\n")

    if count == 0:
        fh.write("The PEP 8 checker didn't find any issues.\n")
        fh.close()
        continue

    fh.write("\n")
    fh.write(".. rubric:: Statistic\n")
    fh.write("\n")
    fh.write("======= ======================================================\n")
    fh.write("Count   PEP 8 message string                                  \n")
    fh.write("======= ======================================================\n")
    for stat in statistic:
        fh.write(stat + "\n")
    fh.write("======= ======================================================\n")
    fh.write("\n")

    fh.write(".. rubric:: Warnings\n")
    fh.write("\n")
    fh.write("::\n")
    fh.write("\n")

    data = data.replace(path, '    obspy')
    fh.write(data)
    fh.write("\n")

    fh.close()

# restore stdout
sys.stdout = stdout
