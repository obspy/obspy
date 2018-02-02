# WARNING: this script will remove obspy related docker images on a regular basis
#
# this script is currently used in a Debian Linux virtual machine's crontab
# (which gets started headless once per day):
# 
# # a personal github authorization token with "repo:status" OAuth scope has to be provided to work around anonymous github API limits
# OBSPY_COMMIT_STATUS_TOKEN=....
# @reboot mv $HOME/docker_testbot.log $HOME/docker_testbot.log.old; cd /path/to/dedicated/obspy/checkout/misc/docker/ && bash ./cronjob_docker_tests.sh -t -d -b -p > $HOME/docker_testbot.log 2>&1

# path to a dedicated obspy clone for running docker tests,
# we run a lot of radical `git clean -fdx` in that repo, so any local changes
# will be regularly lost
# obspy main repo is expected to be registered as remote "origin"
if [ -z "$OBSPY_DOCKER_BASE" ]
then
    echo "Env variable OBSPY_DOCKER_BASE must be set to the root directory of a dedicated ObsPy git clone (Warning: 'git clean -fdx' will be run!)."
    exit 1
fi
OBSPY_DOCKER=$OBSPY_DOCKER_BASE/misc/docker

DOCKER_TESTS=false
DOCKER_DEB_PACKAGING=false
WORK_ON_BRANCHES="None"
WORK_ON_PRS="False"
# Process command line arguments
#  Specifying what type of docker jobs to execute..
#     -t    Run docker tests
#     -d    Run docker deb packaging/testing
#  ..and on which targets to operate:
#     -b    Work on main branches (master, maintenance_1.0.x, ..)
#     -p    Work on open pull requests
while getopts tdbp opt
do
   case "$opt" in
      t) DOCKER_TESTS=true;;
      d) DOCKER_DEB_PACKAGING=true;;
      b) WORK_ON_BRANCHES="['master', 'maintenance_1.0.x']";;
      p) WORK_ON_PRS="True";;
   esac
done


echo "##### Start of obspy docker cronjob at $(date)"

# The following is only useful/safe if the docker testbot is set up inside a virtual machine
# XXX # sleep for some time, so that the cronjob can be killed if the VM is entered
# XXX # for some manual tampering
# XXX sleep 300
# XXX # keep VM up to date
# XXX sudo aptitude update && sudo aptitude upgrade -y

DOCKER=`which docker.io || which docker`

# start with a clean slate, remove all cached docker containers
# The following is only useful/safe if the docker testbot is set up inside a virtual machine
# XXX $DOCKER rm $($DOCKER ps -qa)
# only remove images if older than 7 days. data for the images is over 20 GB so
# we do not want to recreate them on a hourly basis.. ;-)
DOCKER_IMAGE_AGE_THRESHOLD=`date -Ins --date='7 days ago'`
# only remove images explicitly listed as used by obspy
DOCKER_IMAGES=`cat $OBSPY_DOCKER/obspy_images`
echo "##### Checking age of Docker images"
echo "Will remove obspy related images older than $DOCKER_IMAGE_AGE_THRESHOLD"
for DOCKER_IMAGE in $DOCKER_IMAGES
do
    echo "## $DOCKER_IMAGE"
    DOCKER_ID=$($DOCKER images $DOCKER_IMAGE -q)
    # image not listed, so just skip, it will be created from scratch anyway
    if [[ ! $DOCKER_ID ]]
    then
        continue
    fi
    IMAGE_AGE=$($DOCKER inspect --format='{{.Created}}' --type=image $DOCKER_ID)
    echo "Found image for $DOCKER_IMAGE, ID: $DOCKER_ID, created: $IMAGE_AGE"
    if [[ "$IMAGE_AGE" < "$AGE_THRESHOLD" ]]
    then
        echo "Removing outdated docker image $DOCKER_IMAGE $DOCKER_ID"
        $DOCKER rmi $DOCKER_ID
    fi
done


# we don't want to operate in "cp" mode of the docker test script..
export OBSPY_DOCKER_TEST_SOURCE_TREE="clone"

cd $OBSPY_DOCKER_BASE || exit 1
git fetch origin  # this assumes "origin" is the remote alias for the obspy/obspy main repository!
git clean -fdx
git checkout master  # "master" should be set up to track "master" branch of obspy/obspy
git pull
git reset --hard FETCH_HEAD
git clean -fdx
cd $OBSPY_DOCKER
# check version number of obspy_github_api
python -c 'from obspy_github_api import __version__; assert [int(x) for x in __version__.split(".")[:2]] >= [0, 5]' || exit 1

# run docker tests if requested:
cd ${OBSPY_DOCKER}
if [ "$DOCKER_TESTS" = true ]
then
    TARGETS=`python -c "from obspy_github_api import get_docker_build_targets; print(get_docker_build_targets(context='docker-testbot', branches=${WORK_ON_BRANCHES}, prs=${WORK_ON_PRS}))"`
    TARGET_ARRAY=(${TARGETS// / })
    echo "##### DOCKER TESTS, ALL TARGETS (${#TARGET_ARRAY[@]}): ${TARGETS}"
    for TARGET in $TARGETS
    do
        echo "##### DOCKER TESTS, WORKING ON TARGET: ${TARGET}"
        date
        # only allowed special character in github user names is dash ('-'),
        # so we use underscore as a separator after the issue number
        # TARGET is e.g. 1397_andres-h:3f9d48fdaad19051e7f8993dc81119f379d1041b
        PR_REPO_GITTARGET=(${TARGET//_/ })
        PR=${PR_REPO_GITTARGET[0]}
        REPO_GITTARGET=${PR_REPO_GITTARGET[1]}
        echo "##### RUNNING DOCKER TESTS FOR TARGET: ${REPO_GITTARGET}"
        if [ "$PR" = "XXX" ]
        then
            bash run_obspy_tests.sh -s -t${REPO_GITTARGET} -e"--all"
        else
            bash run_obspy_tests.sh -s -t${REPO_GITTARGET} -e"--pr-url=https://github.com/obspy/obspy/pull/${PR}"
        fi
    done
fi

# run docker deb packaging+testing if requested:
cd ${OBSPY_DOCKER}
if [ "$DOCKER_DEB_PACKAGING" = true ]
then
    TARGETS=`python -c "from obspy_github_api import get_docker_build_targets; print(get_docker_build_targets(context='docker-deb-buildbot', branches=${WORK_ON_BRANCHES}, prs=${WORK_ON_PRS}))"`
    TARGET_ARRAY=(${TARGETS// / })
    echo "##### DOCKER DEB PACKAGING, ALL TARGETS (${#TARGET_ARRAY[@]}): ${TARGETS}"
    for TARGET in $TARGETS
    do
        echo "##### DOCKER DEB PACKAGING, WORKING ON TARGET: ${TARGET}"
        date
        # only allowed special character in github user names is dash ('-'),
        # so we use underscore as a separator after the issue number
        # TARGET is e.g. 1397_andres-h:3f9d48fdaad19051e7f8993dc81119f379d1041b
        PR_REPO_GITTARGET=(${TARGET//_/ })
        PR=${PR_REPO_GITTARGET[0]}
        REPO_GITTARGET=${PR_REPO_GITTARGET[1]}
        echo "##### DOCKER DEB PACKAGING, PACKAGING AND RUNNING TESTS FOR TARGET: ${REPO_GITTARGET}"
        bash package_debs.sh -s -t${REPO_GITTARGET}
    done
fi


# The following is only useful/safe if the docker testbot is set up inside a virtual machine
# XXX # sleep for some time, so a login user has a chance to kill the cronjob before
# XXX # it halts the VM
# XXX (sleep 600; sudo halt -p)
