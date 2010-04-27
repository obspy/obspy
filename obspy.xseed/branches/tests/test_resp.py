# -*- coding: utf-8 -*-
"""
Conversion test suite for Dataless SEED into SEED RESP files.

Runs tests against all Dataless SEED files within the data/dataless directory. 
Output is created within the output/resp folder. Once generated files will
be skipped. Clear the output/resp folder in order to rerun all tests.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.xseed import Parser
import glob
import os


# paths
dataless_path = os.path.join("data", "dataless")
resp_path = os.path.join("data", "resp")
output_path = os.path.join("output", "resp")

# generate output directory 
if not os.path.isdir(output_path):
    os.mkdir(output_path)


def _compareRESPFiles(original, new):
    """
    Compares two RESP files.
    """
    org_list = open(original, 'r').readlines()
    new_list = open(new, 'r').readlines()
    # Skip the first line.
    for _i in xrange(1, len(org_list)):
        try:
            assert new_list[_i] == org_list[_i]
        except:
            # Skip if it is the header.
            if org_list[_i] == '#\t\t<< IRIS SEED Reader, Release 4.8 >>\n' and\
               new_list[_i] == '#\t\t<< obspy.xseed, Version 0.1.3 >>\n':
                continue
            # Skip if its a short time string.
            line1 = min(new_list[_i].strip(), org_list[_i].strip())
            line2 = max(new_list[_i].strip(), org_list[_i].strip())
            if line1.startswith('B052F22') and line1 in line2:
                continue
            if line1.startswith('B052F23') and line1 in line2:
                continue
            # search for floating point errors
            diffs = [i for i, c in enumerate(zip(org_list[_i], new_list[_i])) \
                     if c[0] != c[1]]
            if len(diffs) == 1 and diffs[0] != len(org_list[_i]):
                if new_list[_i][diffs[0] + 1] == 'E' and \
                   new_list[_i][diffs[0]].isdigit():
                    continue
            msg = os.linesep + 'Compare failed for:' + os.linesep + \
                  'File :\t' + original.split(os.sep)[-1] + \
                  '\nLine :\t' + str(_i + 1) + os.linesep + \
                  'EXPECTED:' + os.linesep + org_list[_i] + \
                  'GOT:' + os.linesep + new_list[_i]
            raise AssertionError(msg)

# build up file list and loop over all files
files = []
files += glob.glob(os.path.join(dataless_path, '*', '*'))
files += glob.glob(os.path.join(dataless_path, '*', '*', '*'))
for file in files:
    # skip arclink
    #if 'arclink' in file:
    #    continue
    # check and eventually generate output directory
    path = os.path.dirname(file)
    relpath = os.path.relpath(path, dataless_path)
    path = os.path.join(output_path, relpath)
    if not os.path.isdir(path):
        os.mkdir(path)
    # skip directories
    if not os.path.isfile(file):
        continue
    # create folder from filename
    seedfile = os.path.basename(file)
    resp_path = os.path.join(path, seedfile)
    # skip existing directories
    if os.path.isdir(resp_path):
        print "Skipping", os.path.join(relpath, seedfile)
        continue
    else:
        os.mkdir(resp_path)
        print "Parsing %s\t\t" % os.path.join(relpath, seedfile)
    # Create the RESP file.
    try:
        sp = Parser(file)
        sp.writeRESP(folder=resp_path)
        sp.writeRESP(folder=resp_path, zipped=True)
        # Compare with RESP files generated with rdseed from IRIS if existing
        for resp_file in glob.iglob(resp_path + os.sep + '*'):
            print '  ' + os.path.basename(resp_file)
            org_resp_file = resp_file.replace('output' + os.sep,
                                              'data' + os.sep)
            if os.path.exists(org_resp_file):
                _compareRESPFiles(org_resp_file, resp_file)
    except Exception, e:
        # remove all related files
        if os.path.isdir(resp_path):
            for f in glob.glob(os.path.join(resp_path, '*')):
                os.remove(f)
            os.removedirs(resp_path)
        # raise actual exception
        raise
