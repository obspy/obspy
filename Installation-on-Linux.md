## Overview
On Linux there is a variety of possible ways to install ObsPy. The overview is given here, please read below for details.

 * Linux package management (only Debian/Ubuntu)
  * latest stable release
  * admin privileges necessary
  * easiest/cleanest way to handle for end user
  * system-wide for all users
  * all dependencies installed autometically
  * automatically notifies of new releases
 * Python Package Index
  * latest stable release _or_ older versions _or_ current developer snapshot
  * either as admin _or_ in a local user environment
 * Using a git checkout of the source code repository
  * current developer snapshot
  * only recommended for experienced users / developers
  * makes it possible to modify internal ObsPy code

## Via Linux Package Management (Debian/Ubuntu)
The easiest and cleanest way to install the most recent stable release ObsPy under Debian and Ubuntu globally for all system users is using the ObsPy Debian Packages. The package management system will automatically install all necessary components and dependencies needed for using ObsPy. The package management will also automatically check for new releases and give an update dialog in case new tagged versions are 

* [[Installation using the ObsPy apt repository|Installation on Linux via Apt Repository]]

## Prepackaged Modules from Python Package Index
On any Linux distribution, prepackaged ObsPy modules for the latest stable release can be installed from the [Python Package Index (PyPI)](http://pypi.python.org/pypi). Furthermore also older version releases and the current developer snapshot can be installed in this way.

* [[Installation via PyPI|Installation on Linux via PyPI]]

## Manually from Source Code
Installing from a git checkout of the source code repository is only recommended for advanced users and developers. It makes it possible to work on modified ObsPy code and on bug fixes.

* [[Installation via PyPI|Installation on Linux from Source]]