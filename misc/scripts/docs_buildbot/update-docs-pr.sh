#!/bin/bash

# check if a PR build is already running
test -f $HOME/update-docs-pr.pid && echo "PR docs checking/building aborted: pid file exists" && exit 1

cd $HOME
anaconda3/bin/python update_pull_request_metadata.py

# check free space and free inodes (current docs server has relatively low
# inode quota)
## `df` on docs server doesn't have "output" option
## FREE_INODES=`df . --output="iavail" | tail -1`
FREE_INODES=`df $HOME/htdocs -i | tail -1 | awk '{print $4}'`
INODE_CRITICAL="300000"  # 300k
if [ "$FREE_INODES" -lt "$INODE_CRITICAL" ]
then
    echo "Aborting PR docs build, low inodes: $FREE_INODES"
    exit 1
fi
FREE_SPACE=`df $HOME/htdocs/ | tail -1 | awk '{print $4}'`
SPACE_CRITICAL="10000000"  # 10G
if [ "$FREE_SPACE" -lt "$SPACE_CRITICAL" ]
then
    echo "Aborting PR docs build, low disk space: $FREE_INODES"
    exit 1
fi


for FILE in `ls pull_request_docs/*.todo 2> /dev/null`
do
    PR=${FILE##*/}
    PR=${PR%.*}
    # for the "<pr-number>.done" file, use the time of the "<pr-number>.todo"
    # file, which is set to one second after the "<pr-number>" file (that
    # reflects the time of the commit). This should make sure that the "done"
    # status correctly reflects the build.
    DATETIME=`stat -c '%y' $FILE`
    echo "Starting docs build for PR $PR"
    bash $HOME/update-docs.sh -p $PR
    STATUS=`echo $?`
    echo "Docs build for PR $PR exited with status: $STATUS"
    if [ ! "$STATUS" = "-1" ]
    then
        # double check that output is really where it should be before marking
        # build as done (in case the docs build gets terminated from outside
        # leading to a non-"-1" return code)
        if [ -f $HOME/update-docs-pr/src/obspy/misc/docs/build/html/index.html ]
        then
            touch -d "$DATETIME" $HOME/pull_request_docs/${PR}.done
            rm $HOME/pull_request_docs/${PR}.todo
        fi
    fi
    cp $HOME/update-docs-pr/log.txt $HOME/pull_request_docs/${PR}.log
    cp $HOME/update-docs-pr/log.txt $HOME/htdocs/docs/pull_requests/${PR}.log
done
