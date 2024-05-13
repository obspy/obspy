#!/bin/bash
# assumes that html docs are built and ready in build/html/

# exit immediately on any error
set -e

DOCSETVERSION=`python -c "import obspy; print(obspy.__version__)"`

cd build/

# this should equally work to get the version number but it seems the "dirty" part of the version numbering is missing there
#DOCSETVERSION=`head -3 build/html/objects.inv | grep 'Version' | awk '{print $NF}'))

doc2dash --verbose --name "ObsPy ${DOCSETVERSION}" --icon html/_static/obspy_logo_no_text_32x32.png --online-redirect-url https://docs.obspy.org/ html/
sed 's#\(<string>[Oo]bs[Pp]y\) .*\(</string>\)#\1 ${DOCSETVERSION}\2#' --in-place "build/ObsPy ${DOCSETVERSION}.docset/Contents/Info.plist"
cat ../docset_css_fixes.css >> "ObsPy ${DOCSETVERSION}.docset/Contents/Resources/Documents/_static/css/custom.css"
if [[ "${DOCSETVERSION}" == *"post"* ]]
then
  DOCSETNAME="master"
else
  DOCSETNAME=$(DOCSETVERSION)
fi

tar --exclude='.DS_Store' -cvzf "obspy-${DOCSETNAME}.docset.tgz" "ObsPy ${DOCSETVERSION}.docset"; \
