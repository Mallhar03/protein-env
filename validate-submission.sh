#!/bin/bash

# OpenEnv Submission Validation Script
# Usage: ./validate-submission.sh [URL]

URL=$1

if [ -z "$URL" ]; then
    echo "Error: No URL provided."
    echo "Usage: ./validate-submission.sh https://your-space-url.hf.space"
    exit 1
fi

echo "--- VALIDATING OPENENV SUBMISSION ---"
echo "Target URL: $URL"

# 1. Check if openenv-core is installed
if ! command -v openenv &> /dev/null; then
    echo "Error: openenv-core CLI not found. Please install it with 'pip install openenv-core'."
    exit 1
fi

# 2. Run openenv validate
echo "Executing: openenv validate --url $URL"
openenv validate --url "$URL"

VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo "--- VALIDATION PASSED ---"
else
    echo "--- VALIDATION FAILED ---"
fi

exit $VALIDATION_EXIT_CODE
