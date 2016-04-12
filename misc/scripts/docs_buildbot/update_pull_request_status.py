import os
import sys
import requests

# github API token with "repo.status" access right
token = os.environ["OBSPY_COMMIT_STATUS_TOKEN"]

commit = sys.argv[1]
status = sys.argv[2]
target_url = sys.argv[3]

if status == "success":
    description = "Check out Pull Request docs build here:"
elif status in ["error", "failure"]:
    description = "Log for failed Pull Request docs build here:"
else:
    raise ValueError("Invalid status: {}".format(status))

url = "https://api.github.com/repos/obspy/obspy/statuses/{}".format(commit)
headers = {"Authorization": "token {}".format(token)}
data = {"state": status, "context": "docs-buildbot",
        "description": description, "target_url": target_url}
r = requests.post(url, json=data, headers=headers)

try:
    assert r.ok
except:
    print(r.json())
