#!/bin/bash

if [ "$(id -un)" != "obspy" ]; then
echo "This script must be run as user obspy!" 1>&2
exit -1
fi

PIDFILE=$HOME/update-docs.pid
GITFORK=obspy
GITTARGET=master
CONDABASE=$HOME/anaconda3
CONDAMAINBIN=$CONDABASE/bin
CONDATMPNAME=tmp_build_docs
BASEDIR=$HOME/update-docs
DOCSBASEDIR=$HOME/htdocs/docs
DOCSNAME=obspy-${GITTARGET}-documentation

# Process command line arguments
while getopts f:t:p:dr opt
do
   case "$opt" in
      f) GITFORK=$OPTARG;;
      t) GITTARGET=$OPTARG;;
      d) BUILD_DOCSET=true;;
      r) RUN_TESTS=true;;
# option -p will override options -t and -f !
      p) BASEDIR=${BASEDIR}-pr
         CONDATMPNAME=${CONDATMPNAME}-pr
         PIDFILE=$HOME/update-docs-pr.pid
         PR_NUMBER=$OPTARG
         FORKTARGET=($(cat $HOME/pull_request_docs/$OPTARG))
         DOCSBASEDIR=${DOCSBASEDIR}/pull_requests
         DOCSNAME=${OPTARG}
         BUILD_PR=true
         ;;
   esac
done

# determine fork and branch from the PR file, overrides options -t/-f
if [ "$BUILD_PR" = true ] ; then
    GITFORK=${FORKTARGET[0]}
    GITTARGET=${FORKTARGET[1]}
fi

CONDATMPBIN=$CONDABASE/envs/$CONDATMPNAME/bin
LOG=$BASEDIR/log.txt
GITDIR=$BASEDIR/src/obspy
DOCSDIR=$DOCSBASEDIR/$DOCSNAME
DOCSETDIR=$HOME/htdocs/docsets
DOCSETNAME="ObsPy ${GITTARGET}.docset"
DOCSET=$DOCSETDIR/$DOCSETNAME

# check if script is already running
test -f $PIDFILE && echo "doc building aborted: pid file exists" && exit -1
# otherwise create pid file
echo $! > $PIDFILE

# set trap to remove pid file after exit of script
function cleanup {
rm -f $PIDFILE
}
trap cleanup EXIT

# clean directory
rm -rf $BASEDIR
mkdir -p $BASEDIR
mkdir -p $GITDIR

# from now on all output to log file
exec > $LOG 2>&1
cd $HOME

# set up build env
. $CONDAMAINBIN/deactivate
$CONDAMAINBIN/conda env remove -y -n $CONDATMPNAME
$CONDAMAINBIN/conda create -y -n $CONDATMPNAME --clone py3-docs-master_mpl202
. $CONDAMAINBIN/activate $CONDATMPNAME

# clone github repository
git clone https://github.com/${GITFORK}/obspy.git $GITDIR

# checkout the state we want to work on
echo "#### Working on $GITTARGET"
cd $GITDIR
git clean -fxd
git checkout -- .
git checkout $GITTARGET
COMMIT=`git rev-parse HEAD`
git clean -fxd

# create github pull request status
if [ "$BUILD_PR" = true ] ; then
    python -c "from obspy_github_api import set_commit_status; set_commit_status(commit='${COMMIT}', status='pending', context='docs-buildbot', description='Docs build started..', target_url='http://docs.obspy.org/pull-requests/pull_request_docs.log')"
fi

if [ "$GITFORK" != "obspy" ]
then
    git remote add upstream git://github.com/obspy/obspy.git
    git fetch upstream
fi

# run develop.sh
cd $GITDIR
git status
git log -1 | cat
# export LDFLAGS="-lgcov"  # coverage on C code (make c_coverage)
#$BASEDIR/bin/python setup.py develop --verbose
python setup.py develop --verbose || exit 1

# keep some packages up to date
# $BASEDIR/bin/pip install --upgrade pip
# $BASEDIR/bin/pip install --upgrade --no-deps pep8==1.5.7 flake8
# $BASEDIR/bin/pip install pyimgur

# make docs
cd $GITDIR/misc/docs
make clean
# make coverage
# "make html" has to run twice
# - before latexpdf (otherwise .hires.png images are not built)
# - after latexpdf (so that the tutorial pdf is included as downloadable file in html docs)
make html && SUCCESS=true || SUCCESS=false
# in addition to return value of `make html` check if index.html exists
if [ ! -f build/html/index.html ] ; then
    SUCCESS=false
fi

CLEAN_SUCCESS=$SUCCESS
if [ "$SUCCESS" = true ] ; then
    grep -i -e Warning -e Error build/html/sphinx-build_warnings.txt && CLEAN_SUCCESS=false
fi

if [ "$SUCCESS" = true ] ; then
    # perform rest of docs building
    if [ "$BUILD_DOCSET" = true ] ; then
        make docset_after_html DOCSETVERSION="$GITTARGET"
    fi
    make latexpdf-png-images
    make html
    make linkcheck
    # make doctest
    # don't do C coverage for PR builds, it needs to run tests and it seems
    # this hangs sometimes, and we don't need it for PR builds
    if [ "$BUILD_PR" = false ] ; then
        make c_coverage
    fi

    # pack build directory
    # move to htdocs
    cd $BASEDIR
    cp $LOG $GITDIR/misc/docs/build/html
    cd $DOCSBASEDIR
    rm -rf $DOCSDIR ${DOCSNAME}.tgz
    cp -a $GITDIR/misc/docs/build/html $DOCSDIR
    cp $GITDIR/misc/docs/build/linkcheck/output.txt $DOCSDIR/linkcheck.txt
    # don't tar the result if we're building a pull request
    if [ "$BUILD_PR" = false ] ; then
        tar -czf ${DOCSNAME}.tgz ${DOCSDIR}
    fi

    # copy docset and rename
    if [ "$BUILD_DOCSET" = true ] ; then
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
            OBSPY_VERSION=`python -c 'import obspy; print(obspy.__version__)'`
            sed "s#<version>.*</version>#<version>${OBSPY_VERSION}</version>#" --in-place obspy-master.xml
        fi
    fi
fi

# create github pull request status
if [ "$BUILD_PR" = true ] ; then
    if [ "$CLEAN_SUCCESS" = true ] ; then
        python -c "from obspy_github_api import set_commit_status; set_commit_status(commit='$COMMIT', status='success', context='docs-buildbot', description='Check out Pull Request docs build here:', target_url='http://docs.obspy.org/pull-requests/${PR_NUMBER}/')"
    elif [ "$SUCCESS" = true ] ; then
        python -c "from obspy_github_api import set_commit_status; set_commit_status(commit='$COMMIT', status='failure', context='docs-buildbot', description='Build succeeded, but there are warnings/errors:', target_url='http://docs.obspy.org/pull-requests/${PR_NUMBER}/build_status.html')"
    else
        python -c "from obspy_github_api import set_commit_status; set_commit_status(commit='$COMMIT', status='error', context='docs-buildbot', description='Log for failed Pull Request docs build here:', target_url='http://docs.obspy.org/pull-requests/${PR_NUMBER}.log')"
    fi
fi

# report (if not in a pull request)
if [ "$RUN_TESTS" = true ] ; then
    $CONDATMPBIN/obspy-runtests -v -x seishub -n sphinx -r --all --keep-images
fi

if [ "$SUCCESS" = false ] ; then
    exit 1
fi
exit 0
