# -*- coding: utf-8 -*-
"""
USAGE: make_c_coverage.py output_dir
"""

from os import walk, rename, makedirs
import sys
from os.path import join, exists, dirname, abspath, pardir, sep
from subprocess import call
from fnmatch import fnmatch
import tempfile
import shutil
from lxml.html import fromstring, tostring, Element

try:
    target_dir = sys.argv[1]
except IndexError:
    raise SystemExit(__doc__)
shutil.rmtree(target_dir, ignore_errors=True)
obspy_dir = abspath(join(dirname(__file__), pardir, pardir))
build_dir = join(obspy_dir, 'build')
kwargs = {'shell': True, 'cwd': obspy_dir}


# GENERATE COVERAGE
shutil.rmtree(build_dir, ignore_errors=True)
cflags = 'CFLAGS="-O0 -fprofile-arcs -ftest-coverage -coverage"'
cmd = cflags + ' python setup.py develop'
call(cmd, **kwargs)
call('obspy-runtests -d', **kwargs)


# FIND ALL COVERAGE PROFILE STATISTICS
profs = []
for root, dirs, files in walk(build_dir):
    profs.extend(abspath(join(root, i)) for i in files if fnmatch(i, '*.gcda'))

# GENERATE REPORTS WITH GCOV
cov = []
for gcda in profs:
    source = gcda[gcda.rfind('obspy' + sep):].replace('gcda', 'c')
    if not exists(join(obspy_dir, source)):
        source = source.replace('.c', '.f')
    with tempfile.NamedTemporaryFile() as fp:
        cmd = 'gcov --object-file %s %s' % (gcda, source)
        call(cmd, stdout=fp, **kwargs)
        fp.seek(0)
        # read stdout
        filename = fp.readline().strip().split()[1].strip("'")
        perc = float(fp.readline().split(':')[1].split('%')[0])
        gcov = fp.readline().strip().split()[1].strip("'")
        # move genereted gcov to coverage folder
        new_dir = join(target_dir, dirname(source))
        try:
            makedirs(new_dir)
        except OSError:
            pass
        rename(join(obspy_dir, gcov), join(new_dir, gcov))
        cov.append((filename, join(new_dir, gcov), perc))


# GENERATE HTML
page = fromstring("<html><table></table></html>")
table = page.xpath('.//table')[0]
for name, gcov, perc in cov:
    td1, td2 = Element('td'), Element('td')
    gcov = gcov.replace(target_dir, '.')
    a = Element('a', attrib={'href': gcov})
    a.text = name
    td1.append(a)
    td2.text = "%6.2f%%" % perc
    tr = Element('tr')
    tr.extend([td1, td2])
    table.append(tr)
with open(join(target_dir, 'index.html'), 'wb') as fp:
    fp.write(tostring(page))
