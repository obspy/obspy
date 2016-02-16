# -*- coding: utf-8 -*-
"""
USAGE: make_c_coverage.py output_dir
"""

import os
import shutil
import subprocess
import sys
from fnmatch import fnmatch

from lxml.html import Element, fromstring, tostring


try:
    target_dir = sys.argv[1]
except IndexError:
    raise SystemExit(__doc__)
obspy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         os.path.pardir, os.path.pardir))
build_dir = os.path.join(obspy_dir, 'build')


def cleanup(wildcard='*.o'):
    for root, dirs, files in os.walk(obspy_dir):
        for f in files:
            if fnmatch(f, wildcard):
                os.unlink(os.path.join(root, f))


# cleanup to force rebuild, python setup.py clean --all develop does not
# force a rebuild of all files, therefore manually cleaning up here
shutil.rmtree(target_dir, ignore_errors=True)
shutil.rmtree(build_dir, ignore_errors=True)
cleanup('*.o')
cleanup('*.so')

# GENERATE COVERAGE
os.environ['CFLAGS'] = "-O0 -fprofile-arcs -ftest-coverage"
os.environ['FFLAGS'] = "-O0 -fprofile-arcs -ftest-coverage -fPIC"
os.environ['OBSPY_C_COVERAGE'] = "True"
subprocess.call([sys.executable, 'setup.py', '-v', 'develop'], cwd=obspy_dir)
subprocess.call(['obspy-runtests', '-d'], cwd=obspy_dir)


# FIND ALL COVERAGE PROFILE STATISTICS
profs = []
for root, dirs, files in os.walk(build_dir):
    profs.extend(os.path.abspath(os.path.join(root, i))
                 for i in files if fnmatch(i, '*.gcda'))

# GENERATE REPORTS WITH GCOV
cov = []
for gcda in profs:
    source = gcda[gcda.rfind('obspy' + os.path.sep):].replace('gcda', 'c')
    if not os.path.exists(os.path.join(obspy_dir, source)):
        source = source.replace('.c', '.f')

    output = subprocess.check_output(['gcov', '--object-file', gcda, source],
                                     cwd=obspy_dir)
    output = output.decode().splitlines()

    filename = output[0].strip().split()[1].strip("'")
    perc = float(output[1].split(':')[1].split('%')[0])
    gcov = output[2].strip().split()[1].strip("'")

    # move generated gcov to coverage folder
    new_dir = os.path.join(target_dir, os.path.dirname(source))
    try:
        os.makedirs(new_dir)
    except OSError:
        pass
    os.rename(os.path.join(obspy_dir, gcov), os.path.join(new_dir, gcov))
    cov.append((filename, os.path.join(new_dir, gcov), perc))


# GENERATE HTML
page = fromstring("<html><table></table></html>")
table = page.xpath('.//table')[0]
for name, gcov, perc in cov:
    td1, td2 = Element('td'), Element('td')
    gcov = gcov.replace(target_dir, './')
    a = Element('a', attrib={'href': gcov})
    a.text = name
    td1.append(a)
    td2.text = "%6.2f%%" % perc
    tr = Element('tr')
    tr.extend([td1, td2])
    table.append(tr)
with open(os.path.join(target_dir, 'index.html'), 'wb') as fp:
    fp.write(tostring(page))

cleanup('*.o')
