#!/bin/bash

set -e

mkdir /opt/rclone_temp
export RCLONE_DRIVE_SERVICE_ACCOUNT_FILE="/opt/rclone_temp/google-echopype.json"
export RCLONE_DRIVE_ROOT_FOLDER_ID=${ROOT_FOLDER_ID}
export RCLONE_DRIVE_SCOPE=drive
export RCLONE_CONFIG_GDRIVE_TYPE=drive

echo ${GOOGLE_SERVICE_JSON} | jq . > ${RCLONE_DRIVE_SERVICE_ACCOUNT_FILE}

# Little check to make sure we can list from google drive
rclone ls gdrive:

TEST_DATA_FOLDER=${GITHUB_WORKSPACE}/echopype/test_data
if [ -d $TEST_DATA_FOLDER ]
then
    echo "Removing old test data"
    rm -rf $TEST_DATA_FOLDER
    echo "Copying new test data from google drive"
    rclone copy gdrive: $TEST_DATA_FOLDER
    echo "Done"
    
    chmod -R ugoa+w $TEST_DATA_FOLDER
    ls -lah $TEST_DATA_FOLDER
else
    echo "${TEST_DATA_FOLDER} not found"
fi

