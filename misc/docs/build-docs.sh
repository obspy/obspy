#!/bin/bash

if [ "$(id -un)" != "obspy" ]; then
echo "This script must be run as user obspy!" 1>&2
exit 1
fi

GITFORK=obspy
GITTARGET=master
# Process command line arguments
while getopts f:t: opt
do
   case "$opt" in
      f) GITFORK=$OPTARG;;
      t) GITTARGET=$OPTARG;;
   esac
done

BASEDIR=$HOME/update-docs
LOG=$BASEDIR/log.txt
TGZ=$HOME/.backup/update-docs.tgz
GITDIR=$BASEDIR/src/obspy
PIDFILE=$BASEDIR/update-docs.pid
DOCSNAME=obspy-${GITTARGET}-documentation
DOCSBASEDIR=$HOME/htdocs/docs
DOCSDIR=$DOCSBASEDIR/$DOCSNAME
DOCSETDIR=$HOME/htdocs/docsets
DOCSETNAME="ObsPy ${GITTARGET}.docset"
DOCSET=$DOCSETDIR/$DOCSETNAME

# clean directory
rm -rf $BASEDIR
mkdir -p $BASEDIR

# check if script is alread running
test -f $PIDFILE && echo "doc building aborted: pid file exists" && exit 1
# otherwise create pid file
echo $! > $PIDFILE

# from now on all output to log file
exec > $LOG 2>&1

# set trap to remove pid file after exit of script
function cleanup {
rm -f $PIDFILE
}
trap cleanup EXIT
# unpack basedir
cd $HOME
tar -xzf $TGZ

# clone github repository
git clone https://github.com/${GITFORK}/obspy.git $GITDIR

# checkout the state we want to work on
echo "#### Working on $GITTARGET"
cd $GITDIR
git clean -fxd
git checkout -- .
git checkout $GITTARGET
git clean -fxd

if [ "$GITFORK" != "obspy" ]
then
    git remote add upstream git://github.com/obspy/obspy.git
    git fetch upstream
fi

# use unpacked python
export PATH=$BASEDIR/bin:$PATH

# run develop.sh
cd $GITDIR
# export LDFLAGS="-lgcov"  # coverage on C code (make c_coverage)
$BASEDIR/bin/python setup.py develop --verbose

# keep some packages up to date
$BASEDIR/bin/pip install --upgrade pip
$BASEDIR/bin/pip install --upgrade --no-deps pep8==1.5.7 flake8
$BASEDIR/bin/pip install pyimgur

# make docs
cd $GITDIR/misc/docs
make clean
make pep8
# make coverage
# "make html" has to run twice
# - before latexpdf (otherwise .hires.png images are not built)
# - after latexpdf (so that the tutorial pdf is included as downloadable file in html docs)
make html
make docset_after_html DOCSETVERSION="$GITTARGET"
make latexpdf-png-images
make html
make linkcheck
# make doctest
make c_coverage

# pack build directory
# move to htdocs
cd $BASEDIR
ln $LOG $GITDIR/misc/docs/build/html
cd $DOCSBASEDIR
rm -rf $DOCSDIR ${DOCSNAME}.tgz
cp -a $GITDIR/misc/docs/build/html $DOCSDIR
cp $GITDIR/misc/docs/build/linkcheck/output.txt $DOCSDIR/linkcheck.txt
tar -czf ${DOCSNAME}.tgz ${DOCSDIR}
# copy docset and rename
rm -rf "${DOCSET}"
cd $GITDIR/misc/docs/build/
cp -a *.docset "${DOCSET}"
if [ "$GITFORK" == "obspy" ] && [ "$GITTARGET" == "master" ]
then
    cd $DOCSETDIR
    # Add some lines to the CSS to make it more suitable for viewing in
    # Dash/Zeal.
    cat $GITDIR/misc/docs/docset_css_fixes.css >> "$DOCSETNAME/Contents/Resources/Documents/_static/css/custom.css"
    rm -f obspy-master.tgz
    tar --exclude='.DS_Store' -cvzf obspy-master.tgz "$DOCSETNAME"
    OBSPY_VERSION=`$BASEDIR/bin/python -c 'import obspy; print obspy.__version__'`
    sed "s#<version>.*</version>#<version>${OBSPY_VERSION}</version>#" --in-place obspy-master.xml
fi


# report
$BASEDIR/bin/obspy-runtests -v -x seishub -n sphinx -r --all --keep-images
exit 0
