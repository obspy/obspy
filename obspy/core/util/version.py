# -*- coding: utf-8 -*-
# Author: Douglas Creager <dcreager@dcreager.net>
# This file is placed into the public domain.

# Calculates the current version number.  If possible, this is the
# output of “git describe”, modified to conform to the versioning
# scheme that setuptools uses.  If “git describe” returns an error
# (most likely because we're in an unpacked copy of a release tarball,
# rather than in a git working copy), then we fall back on reading the
# contents of the RELEASE-VERSION file.
#
# To use this script, simply import it your setup.py file, and use the
# results of get_git_version() as your package version:
#
# from version import *
#
# setup(
#     version=get_git_version(),
#     .
#     .
#     .
# )
#
# This will automatically update the RELEASE-VERSION file, if
# necessary.  Note that the RELEASE-VERSION file should *not* be
# checked into git; please add it to your top-level .gitignore file.
#
# You'll probably want to distribute the RELEASE-VERSION file in your
# sdist tarballs; to do this, just create a MANIFEST.in file that
# contains the following line:
#
#   include RELEASE-VERSION

# NO IMPORTS FROM OBSPY OR FUTURE IN THIS FILE! (file gets used at
# installation time)
import inspect
import io
import os
import re
from subprocess import STDOUT, CalledProcessError, check_output
import warnings


__all__ = ["get_git_version"]

script_dir = os.path.abspath(os.path.dirname(inspect.getfile(
                                             inspect.currentframe())))
OBSPY_ROOT = os.path.abspath(os.path.join(script_dir, os.pardir,
                                          os.pardir, os.pardir))
VERSION_FILE = os.path.join(OBSPY_ROOT, "obspy", "RELEASE-VERSION")


def call_git_describe(abbrev=10, dirty=True,
                      append_remote_tracking_branch=True):
    try:
        p = check_output(['git', 'rev-parse', '--show-toplevel'],
                         cwd=OBSPY_ROOT, stderr=STDOUT)
        path = p.decode().strip()
    except (OSError, CalledProcessError):
        return None

    if os.path.normpath(path) != OBSPY_ROOT:
        return None

    command = ['git', 'describe', '--abbrev=%d' % abbrev, '--always', '--tags']
    if dirty:
        command.append("--dirty")
    try:
        p = check_output(['git', 'describe', '--dirty', '--abbrev=%d' % abbrev,
                          '--always', '--tags'],
                         cwd=OBSPY_ROOT, stderr=STDOUT)
        line = p.decode().strip()
    except (OSError, CalledProcessError):
        return None

    remote_tracking_branch = None
    if append_remote_tracking_branch:
        try:
            # find out local alias of remote and name of remote tracking branch
            p = check_output(['git', 'branch', '-vv'],
                             cwd=OBSPY_ROOT, stderr=STDOUT)
            remote_info = [line_.rstrip()
                           for line_ in p.decode().splitlines()]
            remote_info = [line_ for line_ in remote_info
                           if line_.startswith('*')][0]
            remote_info = re.sub(r".*? \[([^ :]*).*?\] .*", r"\1", remote_info)
            remote, branch = remote_info.split("/")
            # find out real name of remote
            p = check_output(['git', 'remote', '-v'],
                             cwd=OBSPY_ROOT, stderr=STDOUT)
            stdout = [line_.strip() for line_ in p.decode().splitlines()]
            remote = [line_ for line_ in stdout
                      if line_.startswith(remote)][0].split()[1]
            if remote.startswith("git@github.com:"):
                remote = re.sub(r"git@github.com:(.*?)/.*", r"\1", remote)
            elif remote.startswith("https://github.com/"):
                remote = re.sub(r"https://github.com/(.*?)/.*", r"\1", remote)
            elif remote.startswith("git://github.com"):
                remote = re.sub(r"git://github.com/(.*?)/.*", r"\1", remote)
            else:
                remote = None
            if remote is not None:
                remote_tracking_branch = re.sub(r'[^A-Za-z0-9._-]', r'_',
                                                '%s-%s' % (remote, branch))
        except (IndexError, OSError, ValueError, CalledProcessError):
            pass

    # (this line prevents official releases)
    # should work again now, see #482 and obspy/obspy@b437f31
    if "-" not in line and "." not in line:
        version = "0.0.0.dev+0.g%s" % line
    else:
        parts = line.split('-', 1)
        version = parts[0]
        try:
            modifier = '+' if '.post' in version else '.post+'
            version += modifier + parts[1]
            if remote_tracking_branch is not None:
                version += '.' + remote_tracking_branch
        # IndexError means we are at a release version tag cleanly,
        # add nothing additional
        except IndexError:
            pass
    return version


def read_release_version():
    try:
        with io.open(VERSION_FILE, "rt") as fh:
            version = fh.readline()
        return version.strip()
    except IOError:
        return None


def write_release_version(version):
    with io.open(VERSION_FILE, "wb") as fh:
        fh.write(("%s\n" % version).encode('ascii', 'strict'))


def get_git_version(abbrev=10, dirty=True, append_remote_tracking_branch=True):
    # Read in the version that's currently in RELEASE-VERSION.
    release_version = read_release_version()

    # First try to get the current version using “git describe”.
    version = call_git_describe(
        abbrev, dirty=dirty,
        append_remote_tracking_branch=append_remote_tracking_branch)
    # If that doesn't work, fall back on the value that's in
    # RELEASE-VERSION.
    if version is None:
        version = release_version

    # If we still don't have anything, that's an error.
    if version is None:
        warnings.warn("ObsPy could not determine its version number. Make "
                      "sure it is properly installed. This for example "
                      "happens when installing from a zip archive "
                      "of the ObsPy repository which is not a supported way "
                      "of installing ObsPy.")
        return '0.0.0+archive'

    # pip uses its normalized version number (strict PEP440) instead of our
    # original version number, so we bow to pip and use the normalized version
    # number internally, too, to avoid discrepancies.
    version = _normalize_version(version)

    # If the current version is different from what's in the
    # RELEASE-VERSION file, update the file to be current.
    if version != release_version:
        write_release_version(version)

    # Finally, return the current version.
    return version


def _normalize_version(version):
    """
    Normalize version number string to adhere with PEP440 strictly.
    """
    pattern = (
        r'^[0-9]+?\.[0-9]+?\.[0-9]+?'
        r'((a|b|rc)[0-9]+?)?'
        r'(\.post[0-9]+?)?'
        r'(\.dev[0-9]+?)?$'
    )
    # we have a clean release version or another clean version
    # according to PEP 440
    if re.match(pattern, version):
        return version
    # we have an old-style version (i.e. a git describe string), prepare it for
    # the rest of clean up, i.e. put the '.post+' as separator for the local
    # version number part
    elif '.post' in version:
        version = re.sub(r'-', '+', version, count=1)
    elif re.match(r'^[0-9]+?\.[0-9]+?\.[0-9]+?-[0-9]+?-g[0-9a-z]+?$', version):
        version = re.sub(r'-', '.post+', version, count=1)
    # only adapt local version part right
    version = re.match(r'(.*?\+)(.*)', version)
    # no upper case letters
    local_version = version.group(2).lower()
    # only alphanumeric and "." in local part
    local_version = re.sub(r'[^A-Za-z0-9.]', r'.', local_version)
    version = version.group(1) + local_version
    # make sure there's a "0" after ".post"
    version = re.sub(r'\.post\+', r'.post0+', version)
    return version


if __name__ == "__main__":
    print(get_git_version())
