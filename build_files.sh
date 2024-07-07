#!/bin/bash

apt-get update
apt-get install -y libpq-dev

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Collect static files
python3 manage.py collectstatic --noinput
