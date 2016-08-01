#!/bin/bash

mkdir -p logs
LOG=logs/github_get_all_pr_heads_without_docker-testbot_status.log

rm $LOG

# get a list of commit hashes for open pull requests
DATA=`curl --silent -H "Authorization: token ${OBSPY_COMMIT_STATUS_TOKEN}" --request GET 'https://api.github.com/repos/obspy/obspy/pulls?state=open&sort=updated&direction=desc&per_page=100'`
echo $DATA >> $LOG
echo "" >> $LOG
COMMITS=`echo $DATA | grep -s --only-matching --extended-regexp '"statuses_url": "https://api.github.com/repos/obspy/obspy/statuses/([0-9a-z]{40})",' | sed 's#.*/##' | sed 's#".*##'`
echo $COMMITS >> $LOG
echo "" >> $LOG

# helper function to determine if given commit hash has a status with "docker-testbot" context or not
# returns 0 if no such status exists, i.e. build needed
# returns 1 if such status exists, i.e. no build needed
commit_needs_build() {
    curl --silent --show-error --no-buffer -H "Authorization: token ${OBSPY_COMMIT_STATUS_TOKEN}" --request GET "https://api.github.com/repos/obspy/obspy/commits/$1/status" 2>> $LOG | grep -q -s '"context":[ ]*"docker-testbot"' >> $LOG 2>&1
    if [ $? = 0 ]; then return 1; else return 0; fi
}

for COMMIT in $COMMITS
do
    if commit_needs_build $COMMIT
    then
        echo $COMMIT
        echo $COMMIT >> $LOG
    fi
done
