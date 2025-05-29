#!/bin/bash

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the download script
python3 download_data.py

# Deactivate virtual environment
deactivate

# Log the update
echo "Dataset updated at $(date)" >> update_log.txt 