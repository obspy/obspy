## Installing Dependencies on Mac

### Installing Scipy and Matplotlib on OSX (10.6)

Snow Leopard ships with 64 bit Python 2.6.1 and NumPy already installed. However there currently are no pre-built packages for SciPy and matplotlib which hopefully changes soon, so you have to build them yourself.

Make sure you have the Apple Developer Tools installed that ship with Xcode (http://developer.apple.com/tools/Xcode/). These will give you access to gcc, make and a couple of other tools.

 *   Now basically follow the instructions on this blog: http://blog.hyperjeff.net/?p=160
 *   Also have a look at [[Installation from Source on Linux|Installation-on-Linux-from-source]] 

If you are experiencing any problems read the comments on the blog which are also very helpful.

### Installing on a clean OSX Lion (10.7) install
_(Last edited on August 24th, 2011)_

OSX Lions ships with Python 2.7.1 and NumPy 1.5.1 and an easy way to get a fully working ObsPy installation is to use these two in a virtual environment.

Currently SciPy and Matplotlib will only compile with the repository versions. This will most likely change with new version releases that have adopted to the changes in OSX Lion so they should be easily with pip/easy\_install. SciPy currently also does not work with the standard llvm-gcc of OSX Lion so a different compiler is needed.

#### Preliminaries

Install XCode from the AppStore (this will also install git and some other tools),

  *   http://itunes.apple.com/en/app/xcode/id448457090

Furthermore pkg-config and a Fortran compiler are needed and a very easy way to install both is to use brew as a package manager. It also has a nice way of handling the different packages and not touching the system libraries.

  *   http://mxcl.github.com/homebrew/

You can install further requirements using brew (other ways of installing are of course fine as well):
```
brew install pkg-config
brew install gfortran
```
obspy.taup is linked against libgfortran.a and for some reason it is not found on most systems. You will have to search for libgfortran.a on your system and add the corresponding path to the LIBRARY_PATH variable to be able to link anything against the fortran library using another compiler. I am not sure how this can be avoided (See ticket #262 for more information). Change the path **according to your system** - if you use homebrew, you can use something like this (make sure the version number is correct):

```bash
export LIBRARY_PATH=/usr/local/Cellar/gfortran/4.7.2/gfortran/lib/
```
#### Creating a virtual environment

I strongly recommend using a virtual python environment instead of installing everything to the system python libraries. Newer versions also ship with the pip installer.

 *   Download virtualenv and put it somewhere 

       http://www.virtualenv.org

 *   Create a new virtualenv
     ```
python ~/local/scripts/virtualenv.py --distribute --unzip-setuptools ~/local/python_virtualenvs/system_python_2.7.1
     ```
 *   Activate it (you might want to put this line in your ~/.bash_profile)
     ```
source ~/local/python_virtualenvs/system_python_2.7.1/bin/activate
     ```
#### Installing IPython, SciPy, matplotlib, ..., and ObsPy

By now many dependencies can be installed simply with setuptools/pip but others require more work.

Run the following script with an **ACTIVATED** virtual environment to automatically install all dependencies and (optionally) all ObsPy modules. See the script for more details.

   https://github.com/obspy/obspy/raw/master/misc/scripts/install_dependencies_osx_lion.sh

***

### Other approaches

There are several different other approaches also suitable for older versions of OSX:

 *   The easiest option is to buy a prepacked version from [Enthought](http://www.enthought.com/) ([Read more ...](http://www.enthought.com/products/getepd.php))
    *   check also for free academic licenses ([Read more](http://www.enthought.com/products/edudownload.php)) 
 *   Or to install them via the mac ports ([Read more](http://www.janeriksolem.net/2010/08/installing-numpy-and-matplotlib-on-mac.html)) _(However this did not work with the last version, could be fixed in the current)_
 *   Or to use them from the sage package ([[Read more|Installation on Mac using Sage]])
 *   Or to install them from Source