#!/usr/bin/env sh

#=============================================
# License:
# GNU Lesser General Public License, Version 3
# (http://www.gnu.org/copyleft/lesser.html)
#=============================================

# Script that takes the compiled Cocoa ObsPy OSX Application and the Python
# installation at /Applications/ObsPy.app, merges both and creates a DMG image
# out of them.
#
# The command line image creation requires the paid version of "DMG Canvas" by
# the Araelium Group.
#
# The result of this script is a finalized DMG image called ObsPy.dmg.

rm -rf temp
mkdir temp
cp -pPR ~/Library/Developer/Xcode/DerivedData/ObsPy-*/Build/Products/Debug/ObsPy.app temp

cp -pPR /Applications/ObsPy.app/Contents/MacOS/bin temp/ObsPy.app/Contents/MacOS/
cp -pPR /Applications/ObsPy.app/Contents/MacOS/include temp/ObsPy.app/Contents/MacOS/
cp -pPR /Applications/ObsPy.app/Contents/MacOS/lib temp/ObsPy.app/Contents/MacOS/
cp -pPR /Applications/ObsPy.app/Contents/MacOS/share temp/ObsPy.app/Contents/MacOS/
cp -pPR /Applications/ObsPy.app/Contents/MacOS/Python.framework temp/ObsPy.app/Contents/MacOS/

# Create the dmg image.
dmgcanvas ObsPy.dmgCanvas ObsPy.dmg -v ObsPy App
