Due to the fact that the OSX installation of ObsPy and mainly its dependencies has proven to be problematic, a completely self-contained version of ObsPy and all its dependencies is offered here. Its intention is to facilitate the usage of ObsPy on OSX but the underlying Python installation can of course also be used for other purposes. It currently contains Python 2.7.2 and the [[modules listed here|Installation on Mac with OSX Application: Contents]].

[![OSX install](http://www.obspy.org/osx/thumb_image.png)](http://www.obspy.org/osx/image.png)
[![OSX install](http://www.obspy.org/osx/thumb_app.png)](http://www.obspy.org/osx/app.png)
[![OSX install](http://www.obspy.org/osx/thumb_virtualenv.png)](http://www.obspy.org/osx/virtualenv.png)
[![OSX install](http://www.obspy.org/osx/thumb_update.png)](http://www.obspy.org/osx/update.png)

## Download and Installation

Download the image by clicking on the link in the green box on the right side. Mount the image after the download has completed. Now simply drag the ObsPy.app icon to the Applications folder as you would with every other application.

Installation to another folder will most likely fail due to some fixed paths.

The latest version is available here:
  * [ObsPy-0.1.4.dmg](http://www.obspy.org/osx/ObsPy-0.1.4.dmg) (91.8 MB) ([changelog](http://www.obspy.org/osx/changelog.html), [older versions](http://www.obspy.org/osx))

The Application works with OSX 10.6 and 10.7. It might also work on OSX 10.5 with an Intel processor but that is untested so far.

## Features and Usage

Upon launching, a small window will appear. It gives you immediate access to a fully working IPython console which can be used for ObsPy's interactive features or also to run a Python script.

For a deeper integration in the system run the virtual environment assistant. Make sure to choose an empty folder. One of the available options is to activate the virtual environment every time a new bash terminal is opened by adding it to your `~/.bash_profile`. It has to exist beforehand.

The reinstallations of IPython and readline in the virtual enviroment are necessary because the virtual enviroment is not aware of the existing IPython installation. Readline also needs to be reinstalled for some unknown reason. Virtualenv ships with an old distribute version that is not able to install ObsPy and possibly also other newer packages. Therefore an upgrade is recommended. All options can be disabled.

Open a new terminal after running the assistant and you can, after (automatic) activation of the virtual enviroment, use it like a normal system installation. Check with which to make sure the correct version is called. Use `python/ipython` to run the interpreter or `pip/easy_install` to install additional modules to your virtual environment.

## Updating

The application will periodically check for updates or you can check manually. It is possible to update straight from the Application. This should not break any preexisting virtual environments.

## Deinstallation

Just move the Application to the Trash.