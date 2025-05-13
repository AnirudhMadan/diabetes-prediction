#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate            #. venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
