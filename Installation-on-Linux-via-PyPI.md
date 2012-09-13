It is recommended to install the required dependencies via your package manager beforehand, e.g. on.. 

 * Debian/Ubuntu
```bash
      sudo apt-get update
      sudo apt-get install python
      sudo apt-get install python-dev
      sudo apt-get install python-setuptools
      sudo apt-get install python-numpy
      sudo apt-get install python-scipy
      sudo apt-get install python-matplotlib
      sudo apt-get install python-lxml
      sudo apt-get install ipython         # strongly recommended, not necessary
      #sudo apt-get install python-omniorb # only needed for deprecated module obspy.fissures
```
 * openSUSE
```bash
      sudo zypper update
      sudo zypper install python
      sudo zypper install python-devel
      sudo zypper install python-setuptools
      sudo zypper install python-numpy
      sudo zypper install python-numpy-devel
      sudo zypper install python-scipy
      sudo zypper install python-matplotlib
      sudo zypper install python-matplotlib-tk
      sudo zypper install python-lxml
      sudo zypper install IPython             # strongly recommended, not necessary
      #sudo zypper install python-omniORB     # only needed for deprecated module obspy.fissures
```

First, make sure the version of distribute is recent enough (the current developer version needs at least version 0.6.21):

```bash
easy_install -U distribute
```

To install ObsPy packages run the following commands (see [here](http://pypi.python.org/pypi?%3Aaction=search&term=obspy.&submit=search) for a list of packages available from Python Package Index):

```bash
easy_install -N obspy.core
easy_install -N obspy.mseed
easy_install -N obspy.sac
easy_install -N obspy.gse2
easy_install -N obspy.imaging
...
```

### Notes
 * Packages may be updated to a newer version using the **-U** option:
```bash
easy_install -N -U obspy.core
```
 * **-N**: Option will prevent easy_install to resolve the dependencies on its own (can be useful if dependencies are already installed and installing them via PyPI fails).
 * **==dev**: The latest development version can be obtained by adding ==dev to the package name, e.g. `easy_install -N obspy.sac==dev`.
 * Developers might be interested in directly linking their git checkout to the python site-packages:
   * Clone git repository:
```bash
git clone https://github.com/obspy/obspy.git /path/to/my/obspy
```
   * Install in Python site-packages (will install for current Python interpreter, check with `which python`):
```bash
cd /path/to/my/obspy
cd misc/scripts
./develop.sh
```