### Using the Python environment of Sage in OSX

The open-source mathematics software Sage contains a full Python environment with many pre-installed packages (e.g. NumPy, SciPy, Matplotlib, IPython, ...).

To use it download the Sage binaries for OSX from the Sage homepage:  http://www.sagemath.org/. Follow the installation instructions included with Sage. It is probably a good choice to install it to a directory you do not need root privileges for.

#### The easy way

The easiest way to use IPython is to call the sage script in the Sage root directory with an ipython flag:
```
SAGE_ROOT_DIR/sage -ipython
```
This takes care of all path issues. If you want to be able to call it with ipython you can set an alias:
```
alias ipython='SAGE_ROOT_DIR/sage -ipython'
```
You can also put it into your $HOME/.bash_profile. For an interactive matplotlib plot you might have to follow the instructions below (Interactive matplotlib).

If a cleaner solution is desired keep on reading.

#### The clean way

It is necessary to set the dynamic library path of OSX to the lib directory inside the Sage directory (**in the following replace "SAGE_DIR" with your Sage directory**):
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"SAGE_DIR"/local/lib
```
Then you are able to use all binaries in "SAGE_DIR"/local/bin, e.g. ipython.

If you want permanent access to the binaries, open $HOME/.bash_profile and append the following two lines to the file:
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"SAGE_DIR"/local/lib
export PATH="SAGE_DIR"/local/bin:$PATH
```
Save the file and restart the console.

#### Interactive matplotlib

An interactive matplotlib plot will probably not work by default. Follow [these](http://wiki.sagemath.org/sage_matlab) instructions to make it work. You can always check whether it works or not with the following command which should popup a nice little window:
```
ipython -pylab -c "x=randn(1000); hist(x); show()"
```
If it does not work make sure that the correct Python libraries for the matplotlib backend are used. You can check with
```
ipython -c "import matplotlib.backends; matplotlib.backends"
```
#### Getting ready for ObsPy installation

In order to go on with the usual [[ObsPy installation|Installation-on-Linux-via-PyPI]], you need to install lxml and distribute, the later will overwrite the already installed easy_install script of sage.
```
easy_install distribute
easy_install lxml
```
If this is not working install an virtual environment which points to the sage installation ([[read more|NoRootEasyInstall]]).