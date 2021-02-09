#!/bin/bash

set -e

# ==== ONLY EDIT WITHIN THIS BLOCK =====
BASE_PATH=/usr/local/apache2
DATA_PATH=${BASE_PATH}/htdocs/data

if [ -n "$GOOGLE_SERVICE_JSON" ]
then 
    echo ${GOOGLE_SERVICE_JSON} > "google-echopype.json"
    SERVICE_ACCOUNT_FILE=$(realpath google-echopype.json)
    export RCLONE_DRIVE_SERVICE_ACCOUNT_FILE=${SERVICE_ACCOUNT_FILE}
fi

if [ -n "$TEST_DATA_FOLDER_ID" ]
then 
    export RCLONE_DRIVE_ROOT_FOLDER_ID=${TEST_DATA_FOLDER_ID}
fi
export RCLONE_DRIVE_SCOPE=drive
export RCLONE_CONFIG_GDRIVE_TYPE=drive

rclone copy gdrive: ${DATA_PATH}

# ==== ONLY EDIT WITHIN THIS BLOCK =====

exec "$@"