============================================
License:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)
============================================

This is a collection of scripts and programs to build the ObsPy Application on
OSX.

Requirements:
 * XCode 4 plus developer tools (XCode 3 also works but some script have to be
   adjusted).
 * gfortran compiler
 * git (usually ships with XCode 4)
 * paid version of "DMG Canvas" by Araelium Group (only needed for the
   automatic CLI image creation - the image template can also be manually used
   with the free version)
 * patience

The source code of the Cocoa Application is located in the ObsPy_XCode_Project
folder. The program needs to be compiled in XCode and the create_dmg.sh script
will only work if the resulting program is stored in XCode 4's standard
location. Adjust the script as needed.

The compile_python_and_deps.sh script will download and compile Python, ObsPy
and many other modules to /Applications/ObsPy.app/Contents/MacOS. If the
directory exists it will be deleted by running this script. This path is used
because it is available on every OSX version and installation and the usual
place to install OSX applications. A common path is needed to avoid path issues
and some paths are actually hard coded to make life easier.

The automatic image creation script requires the paid version of "DMG Canvas"
by Araelium Group.  Just run create_dmg.sh after running
compile_python_and_debs.sh and it will create a file ObsPy.dmg in the current
directory that contains the finalized image of the Application ready to be
distributed.

sign_image.sh signs the ObsPy.dmg image in the folder. The resulting signature
is needed for distributing new versions of the application via the Sparkle
framework. The location of the private key has to be stored in the environment
variable OBSPY_PRIVAT_KEY_PATH.

All steps to create and distribute the application/update:
 1. Compile the XCode project from within XCode (just open it and run it).
 2. sh compile_python_and_deps.sh
 3. sh create_dmg.sh
 4. Test everything.
 5. sh sign_image.sh
 6. Upload and adjust obspy_appcast.xml on the server to distribute the update.
