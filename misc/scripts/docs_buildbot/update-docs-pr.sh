#!/bin/bash
cd $HOME
anaconda3/bin/python update_pull_request_metadata.py
for FILE in `ls pull_request_docs/*.todo 2> /dev/null`
do
    PR=${FILE##*/}
    PR=${PR%.*}
    bash $HOME/update-docs.sh -p $PR
    STATUS=`echo $?`
    echo $STATUS
    if [ ! "$STATUS" = "-1" ]
    then
        touch $HOME/pull_request_docs/${PR}.done
        rm $HOME/pull_request_docs/${PR}.todo
    fi
    cp $HOME/update-docs-pr/log.txt $HOME/pull_request_docs/${PR}.log
    cp $HOME/update-docs-pr/log.txt $HOME/htdocs/docs/pull_requests/${PR}.log
done
