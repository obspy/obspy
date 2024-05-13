#!/bin/bash
# assumes that html docs are built and ready in build/html/

# exit immediately on any error
set -e

VERSION=`python -c "import obspy; print(obspy.__version__)"`

cd build/

# this should equally work to get the version number but it seems the "dirty" part of the version numbering is missing there
#DOCSETVERSION=`head -3 build/html/objects.inv | grep 'Version' | awk '{print $NF}'))

doc2dash --name "ObsPy" --icon html/_static/obspy_logo_no_text_32x32.png --online-redirect-url https://docs.obspy.org/ html/  # --verbose
# sed 's#\(<string>[Oo]bs[Pp]y\) .*\(</string>\)#\1 ${VERSION}\2#' --in-place ObsPy.docset/Contents/Info.plist
cat ../docset_css_fixes.css >> ObsPy.docset/Contents/Resources/Documents/_static/css/custom.css
if [[ "${VERSION}" == *"post"* ]]
then
  DOCSETNAME="master"
else
  DOCSETNAME=${VERSION}
fi

tar --exclude='.DS_Store' -cvzf "obspy-${DOCSETNAME}.docset.tgz" ObsPy.docset
