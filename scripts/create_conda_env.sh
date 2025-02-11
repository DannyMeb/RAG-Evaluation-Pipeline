#!/bin/bash

# Last update: Today
# Project: Rag systems evaluation pipeline using Ragas
# Authored by TII/AICCU/Edge-Team
# Final version: Implemented multi-system support (RagFlow, Dify) and improved API handling.

# Script to create a Conda environment and install dependencies from requirements.txt

ENV_NAME="ragas_env"
PYTHON_VERSION="3.10"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda."
    exit
fi

# Create a new Conda environment
echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# Activate the new environment
echo "Activating environment: $ENV_NAME"
source activate $ENV_NAME

# Install the dependencies using requirements.txt (assuming it's in the root directory)
echo "Installing dependencies from requirements.txt"
pip install -r ../requirements.txt

echo "Environment setup completed!"
