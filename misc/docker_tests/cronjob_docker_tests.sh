# this script is currently used in a Debian Linux virtual machine's crontab
# (which gets started headless once per day):
# 
# OBSPY_COMMIT_STATUS_TOKEN=....
# @reboot mv $HOME/docker_testbot.log $HOME/docker_testbot.log.old; bash /path/to/cronjob_docker_tests.sh > $HOME/docker_testbot.log 2>&1

# sleep for some time, so that the cronjob can be killed if the VM is entered
# for some manual tampering
sleep 60
# keep VM up to date
sudo aptitude update && sudo aptitude upgrade -y

# start with a clean slate, remove all cached docker containers and images
docker rm $(docker ps -qa)
docker rmi $(docker images -q)

# path to a dedicated obspy clone for running docker tests,
# we run a lot of radical `git clean -fdx` in that repo, so any local changes
# will be regularly lost
# obspy main repo is expected to be registered as remote "origin"
OBSPY_DOCKER=$HOME/obspy_docker_tests
# we don't want to operate in "cp" mode of the docker test script..
export OBSPY_DOCKER_TEST_SOURCE_TREE="clone"

# run docker tests on any commit that is the tip of a pull request and that
# does not have a docker testbot commit status
cd ${OBSPY_DOCKER} || exit 1
git fetch origin
git clean -fdx
git checkout master
git pull
git reset --hard FETCH_HEAD
git clean -fdx
cd ${OBSPY_DOCKER}/misc/docker_tests/
TARGETS=`bash github_get_all_pr_heads_without_docker-testbot_status.sh`
for TARGET in $TARGETS
do
    # TARGET is e.g. '1002-krischer:966a4a93b3376866c9fade5ded73a7d6e3a20492'
    PR_REPO_SHA=(${TARGET//-/ })
    PR=${PR_REPO_SHA[0]}
    REPO_SHA=${PR_REPO_SHA[1]}
    bash run_obspy_tests.sh -t${REPO_SHA} -e"--pr-url=https://github.com/obspy/obspy/pull/${PR}"
done

# run docker tests on maintenance_1.0.x and master
for BRANCH in maintenance_1.0.x master
do
    cd ${OBSPY_DOCKER} || exit 1
    git fetch origin
    git clean -fdx
    git checkout $BRANCH
    git pull
    git reset --hard FETCH_HEAD
    git clean -fdx
    cd ${OBSPY_DOCKER}/misc/docker_tests/
    bash run_obspy_tests.sh
done

# build and test debian packages
cd $HOME
## run tests in chroots
#bash $HOME/schroot_testrun.sh -f obspy -t master
# build and test all deb packages
# bash ./deb_build_testrun.sh

# sleep for some time, so a login user has a chance to kill the cronjob before
# it halts the VM
(sleep 600; sudo halt)
