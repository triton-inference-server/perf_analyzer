# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/bin/bash -e
# Description: This script triggers a GitLab mirror update and waits for completion
# Usage: ./update_gitlab_mirror.sh <gitlab_access_token> <gitlab_mirror_url>

GITLAB_ACCESS_TOKEN=$1
GITLAB_MIRROR_URL=$2

SYNC_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
curl --fail-with-body --request POST --header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "${GITLAB_MIRROR_URL}"

echo "Mirror update request submitted at ${SYNC_TIME}"

# Poll for completion
# GitLab limits the frequency of mirror updates to once every 5 mins.
# If you trigger an update more frequently, it may not start immediately.
# Make sure sync is finished after sync request was submitted
MAX_RETRIES=15
RETRY_INTERVAL=30
# Convert timestamps to epoch for comparison, timezone agnostic
SYNC_TIME_SECONDS=$(date -d "$SYNC_TIME" +%s)
for i in $(seq 1 $MAX_RETRIES); do
    MIRROR_INFO=$(curl --silent --header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "${GITLAB_MIRROR_URL}")
    MIRROR_STATUS=$(echo "$MIRROR_INFO" | jq -r '.update_status')
    LAST_UPDATE=$(echo "$MIRROR_INFO" | jq -r '.last_update_at')
    LAST_UPDATE_SECONDS=$(date -d "$LAST_UPDATE" +%s)
    if [ "$MIRROR_STATUS" = "finished" ] && [ $LAST_UPDATE_SECONDS -gt $SYNC_TIME_SECONDS ]; then
        echo "Mirror sync successful. Last update: $LAST_UPDATE"
        exit 0
    fi
    echo "Waiting for mirror sync to complete. Attempt $i of $MAX_RETRIES"
    echo "Last update: $LAST_UPDATE"
    sleep $RETRY_INTERVAL
done
echo "Mirror sync failed or timed out"
exit 1
