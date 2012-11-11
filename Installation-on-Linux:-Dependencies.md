## via Package Management
It is recommended to install the required dependencies via the package manager of your Linux distribution, e.g. .. 

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

## Manually from Source
If necessary (e.g. in case your Python is too old), you still have the possibility of compiling your local Python source tree, which is described in the following (or in this automated script: https://github.com/obspy/sandbox/blob/master/buildbots/install_python.sh):

### Python

Download the [Python (e.g. 2.6) tar ball](http://www.python.org/download). Install the
Dependencies: _libreadline5-dev sqlite3 libsqlite3-dev tk8.5 tk8.5-dev tcl8.5 tcl8.5-dev gcc-4.3_ (older tk/tcl libraries plus developer versions do it also). Then do:
```bash
mkdir -p $HOME/local/src
cd $HOME/local/src
tar -xzf /path/to/Python2.6.tgz
cd Python 2.6
./configure --prefix=$HOME/local --enable-unicode=ucs4 && make && make install
export PATH=$HOME/local/bin:$PATH
```

**Note:** Do not forget to set the PATH, all the following commands need to be executed by the new local python. The unicode option is needed as NumPy uses 4 bytes unicode and Python interpreter defaults on 2 bytes unicode.

### easy_install

All the following packages are installed via [Distribute](http://pypi.python.org/pypi/distribute) easy_install. Currently there is no official installer for the  Distribute package. But you may just download and run from command line [the Python script](http://python-distribute.org/distribute_setup.py).
```bash
wget http://nightly.ziade.org/distribute_setup.py
python distribute_setup.py
```

All the following packages are installed via easy_install into your local Python tree. The packages (called eggs) are installed your local python site-packages directory $HOME/local/lib/python2.6/site-packages. All installed packages are listed in $HOME/local/lib/python2.6/site-packages/easy-install.pth. For deinstallation of a package, remove it's entry from the easy_install.pth and delete the corresponding egg file from the site-packages directory.

### NumPy

The installation of [NumPy](http://numpy.scipy.org/) directly via PyPI is often problematic. The better way is to locally compile the package and the local binary via easy\_install to the correct place. Download the NumPy tar ball from http://sourceforge.net/projects/numpy/files.
Dependencies: _fftw3 fftw3-dev libatlas3gf-base libatlas-base-dev libatlas-headers gfortran_ (any other fortran compiler and headers, developer versions of lapack and blas do it also)
```bash
cd $HOME/local/src
tar -xzf /path/to/numpy-1.3.0.tar.gz
cd numpy-1.3.0
python -c 'import setuptools; execfile("setup.py")' bdist_egg
easy_install dist/numpy-1.3.0-py2.6-linux-i686.egg
```

### SciPy
As for NumPy it is better to compile  SciPy locally and install it via easy\_install. Download the SciPy tar ball from http://sourceforge.net/projects/scipy/files.
Dependencies: same as NumPy plus _g++_
```bash
cd $HOME/local/src
tar -xzf /path/to/scipy-7.1.0.tar.gz
cd scipy-7.1.0
python -c 'import setuptools; execfile("setup.py")' bdist_egg
easy_install dist/scipy-0.7.1-py2.6-linux-i686.egg
```

### matplotlib

As for NumPy and SciPy it is less problematic to compile matplotlib locally and install it via easy\_install. Download the tar ball from  http://sourceforge.net/projects/matplotlib/files/matplotlib/.
Dependencies: _libfreetype6 libfreetype6-dev libpng12-0 libpng12-dev zlib1g zlib1g-dev pkg-config_
```bash
cd $HOME/local/src
tar -xzf /path/to/matplotlib-0.99.1.1.tar.gz
cd matplotlib-0.99.1.1
vi setup.cfg # comment wxagg and macosx
python -c 'import setuptools; execfile("setup.py")' bdist_egg
easy_install dist/matplotlib-0.99.1.1_r0-py2.6-linux-i686.egg
```

### lxml, IPython

lxml (2.2.2) and IPython (0.10) are mostly unproblematic to install, just grab and install them from PyPI.
Dependencies: _libxml2 libxml2-dev libxslt1.1 libxslt1-dev zlib1g zlib1g zlib1g-dev_
```bash
easy_install ipython
easy_install lxml
```