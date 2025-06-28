#!/usr/bin/env bash
set -euo pipefail

# Run pre-commit on staged files
STAGED_FILES=$(git diff --cached --name-only)
if [[ -n "$STAGED_FILES" ]]; then
    pre-commit run --files $STAGED_FILES
fi

# Ensure services are torn down on exit
trap 'uv run .ci_helpers/docker/setup-services.py --tear-down' EXIT

uv run .ci_helpers/docker/setup-services.py --deploy

CHANGED_FILES=$(git diff --name-only origin/main...HEAD | tr '\n' ',' | sed 's/,$//')

NUM_WORKERS=${NUM_WORKERS:-2}

uv run .ci_helpers/run-test.py --local \
    --pytest-args="--log-cli-level=WARNING,-vvv,-rx,--numprocesses=${NUM_WORKERS},--max-worker-restart=3,--disable-warnings" \
    --include-cov "$CHANGED_FILES"
