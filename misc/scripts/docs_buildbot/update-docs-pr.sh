#!/bin/bash
cd $HOME
anaconda3/bin/python update_pull_request_metadata.py
for FILE in `ls pull_request_docs/*.todo 2> /dev/null`
do
    PR=${FILE##*/}
    PR=${PR%.*}
    DATETIME=`date --iso-8601=seconds`
    bash $HOME/update-docs.sh -p $PR
    STATUS=`echo $?`
    echo $STATUS
    if [ ! "$STATUS" = "-1" ]
    then
        # double check that output is really where it should be before marking
        # build as done (in case the docs build gets terminated from outside
        # leading to a non-"-1" return code)
        if [ -f $HOME/pull_request_docs/${PR}/index.html ]
        then
            touch -d "$DATETIME" $HOME/pull_request_docs/${PR}.done
            rm $HOME/pull_request_docs/${PR}.todo
        fi
    fi
    cp $HOME/update-docs-pr/log.txt $HOME/pull_request_docs/${PR}.log
    cp $HOME/update-docs-pr/log.txt $HOME/htdocs/docs/pull_requests/${PR}.log
done
