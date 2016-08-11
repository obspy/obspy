#!/bin/bash

mkdir -p logs
LOG=logs/github_get_all_pr_heads_without_docker-testbot_status.log

rm $LOG

# get a list of commit hashes for open pull requests
DATA=`curl --silent -H "Authorization: token ${OBSPY_COMMIT_STATUS_TOKEN}" --request GET 'https://api.github.com/repos/obspy/obspy/pulls?state=open&sort=updated&direction=desc&per_page=100'`
echo $DATA >> $LOG
echo "" >> $LOG
# echo without quotes gets rid of line breaks, which makes regex simpler
# than get rid of spaces and tabs with sed
# than extract all matches with grep, one per line
#    ..assuming the following structure in the returned json:
#      ...
#      "comments_url": "https://api.github.com/repos/obspy/obspy/issues/1452/comments",
#      "statuses_url": "https://api.github.com/repos/obspy/obspy/statuses/dd58fbbc33a8d8ce6f1a76013a5daa9e4a7db72a",
#      "head": {
#        "label": "obspy:docker_rm_old_logs",
#        "ref": "docker_rm_old_logs",
#        "sha": "dd58fbbc33a8d8ce6f1a76013a5daa9e4a7db72a",
#        "user": {
#          "login": "obspy",
#      ...
# finally extract the interesting parts with sed and regex
TARGETS=`echo $DATA | sed 's#\s##g' | grep -s --only-matching --extended-regexp '"comments_url":"[^"]*","statuses_url":"[^"]*","head":{"label":"[^"]*","ref":"[^"]*","sha":"[a-z0-9]{40}"' | sed 's#.*/issues/\([0-9]*\)/comments".*"head":{"label":"\([^:]*\):.*"sha":"\([a-z0-9]\{40\}\)"#\1_\2:\3#'`
# output is PRNUMBER_FORK:SHA
echo $TARGETS >> $LOG
echo "" >> $LOG

# helper function to determine if given commit hash has a status with "docker-testbot" context or not
# returns 0 if no such status exists, i.e. build needed
# returns 1 if such status exists, i.e. no build needed
target_needs_build() {
    # split target which is e.g. 1484_obspy:8fb11420de52392bdb61424f1cfc824a1987a02e
    PR_REPO_SHA=(${1//_/ })
    REPO_SHA=${PR_REPO_SHA[1]}
    REPO_SHA=(${REPO_SHA//:/ })
    SHA=${REPO_SHA[1]}
    curl --silent --show-error --no-buffer -H "Authorization: token ${OBSPY_COMMIT_STATUS_TOKEN}" --request GET "https://api.github.com/repos/obspy/obspy/commits/${SHA}/status" 2>> $LOG | grep -q -s '"context":[ ]*"docker-testbot"' >> $LOG 2>&1
    if [ $? = 0 ]; then return 1; else return 0; fi
}

for TARGET in $TARGETS
do
    if target_needs_build $TARGET
    then
        echo $TARGET
        echo $TARGET >> $LOG
    fi
done
