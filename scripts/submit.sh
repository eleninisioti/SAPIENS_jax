#!/bin/bash

# Specify the directory containing the batch job files
job_dir="/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/jz_scripts/2025_01_11"

# Check if the directory exists
if [ ! -d "$job_dir" ]; then
  echo "Directory '$job_dir' does not exist."
  exit 1
fi

# Loop through each file in the directory
for file in "$job_dir"/*; do
  if [ -f "$file" ]; then
    # Check if the file is a regular file
    echo "Submitting job from file: $file"
    sbatch "$file"  # Submit the job using sbatch
  fi
done