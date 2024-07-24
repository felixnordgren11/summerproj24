#!/bin/bash
# To run: ./script_runner.sh

# Path to the program, specific for this directory.
# Change to correct path if running from another directory.
program_path="../../../bin/sd.f95"

# Check if the program exists
if [ ! -f "$program_path" ]; then
  echo "Program not found: $program_path"
  exit 1
fi


$program_path


