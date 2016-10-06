# WARNING: this script will remove ALL docker images on a regular basis!
#
# this script is currently used in a Debian Linux virtual machine's crontab
# (which gets started headless once per day):
# 
# # a personal github authorization token with "repo:status" OAuth scope has to be provided to work around anonymous github API limits
# OBSPY_COMMIT_STATUS_TOKEN=....
# @reboot mv $HOME/docker_testbot.log $HOME/docker_testbot.log.old; cd /path/to/dedicated/obspy/checkout/misc/docker/ && bash ./cronjob_docker_tests.sh > $HOME/docker_testbot.log 2>&1

echo "##### Start of obspy docker cronjob at $(date)"

# sleep for some time, so that the cronjob can be killed if the VM is entered
# for some manual tampering
sleep 300
# keep VM up to date
sudo aptitude update && sudo aptitude upgrade -y

DOCKER=`which docker.io || which docker`
# path to a dedicated obspy clone for running docker tests,
# we run a lot of radical `git clean -fdx` in that repo, so any local changes
# will be regularly lost
# obspy main repo is expected to be registered as remote "origin"
OBSPY_DOCKER_BASE=$HOME/obspy_docker_tests
OBSPY_DOCKER=$OBSPY_DOCKER_BASE/misc/docker

# start with a clean slate, remove all cached docker containers
$DOCKER rm $($DOCKER ps -qa)
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

# run docker tests on any commit that is the tip of a pull request and that
# does not have a docker testbot commit status (or has status "pending")
cd $OBSPY_DOCKER_BASE || exit 1
git fetch origin  # this assumes "origin" is the remote alias for the obspy/obspy main repository!
git clean -fdx
git checkout master  # "master" should be set up to track "master" branch of obspy/obspy
git pull
git reset --hard FETCH_HEAD
git clean -fdx
cd $OBSPY_DOCKER
TARGETS=`bash github_get_build_targets_without_docker-testbot_status.sh`
# TARGETS also include main branches (currently master and maintenance_1.0.x), if they need a build
for TARGET in $TARGETS
do
    echo "##### DOCKER TESTS, WORKING ON TARGET: ${TARGET}"
    date
    # only allowed special character in github user names is dash ('-'),
    # so we use underscore as a separator after the issue number
    # TARGET is e.g. 1397_andres-h:3f9d48fdaad19051e7f8993dc81119f379d1041b
    PR_REPO_SHA=(${TARGET//_/ })
    PR=${PR_REPO_SHA[0]}
    REPO_SHA=${PR_REPO_SHA[1]}
    echo "##### RUNNING DOCKER TESTS FOR TARGET: ${REPO_SHA}"
    bash run_obspy_tests.sh -t${REPO_SHA} -e"--pr-url=https://github.com/obspy/obspy/pull/${PR}"
done


# build and test debian packages
cd ${OBSPY_DOCKER}/misc/docker/
for TARGET in $TARGETS
do
    echo "##### DOCKER DEB PACKAGING, WORKING ON TARGET: ${TARGET}"
    date
    # only allowed special character in github user names is dash ('-'),
    # so we use underscore as a separator after the issue number
    # TARGET is e.g. 1397_andres-h:3f9d48fdaad19051e7f8993dc81119f379d1041b
    PR_REPO_SHA=(${TARGET//_/ })
    PR=${PR_REPO_SHA[0]}
    REPO_SHA=${PR_REPO_SHA[1]}
    echo "##### DOCKER DEB PACKAGING, PACKAGING AND RUNNING TESTS FOR TARGET: ${REPO_SHA}"
    bash package_debs.sh -t${REPO_SHA}
done


# sleep for some time, so a login user has a chance to kill the cronjob before
# it halts the VM
(sleep 600; sudo halt)
