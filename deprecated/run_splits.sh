#!/bin/bash

# Simple script to run Shrub Tree Grass 55km spatial cross-validation splits
# Usage: ./run_splits.sh [custom_output_directory]

# Set default output directory if not provided
OUTPUT_DIR="${1:-data/features/Shrub_Tree_Grass/55km/custom_skcv_splits}"

# Make sure we're in the right directory (project root)
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Set up the environment
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create necessary directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

poetry env use .venv/bin/python
echo "Running Shrub Tree Grass 55km spatial cross-validation splits..."
echo "Output directory: $OUTPUT_DIR"
echo "Project root: $PROJECT_ROOT"

# Run the Python script
poetry run python3.11 run_shrub_tree_grass_55km_splits.py --output-dir "$OUTPUT_DIR" --overwrite

echo "Split generation completed!"
echo "Results saved to: $OUTPUT_DIR"