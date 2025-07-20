#!/bin/bash

set -e

# Export WandB API key for logging
# export WANDB_API_KEY=

# Set dataset name from argument
DATASET=$1

if [ -z "$DATASET" ]; then
  echo "Usage: $0 [dataset_name]"
  echo "Example: $0 esc50"
  exit 1
fi

# Step 1: Download the dataset (runs dataset.py or dataset.sh)
echo "Checking and downloading dataset: $DATASET"
bash data_download/download_data.sh $DATASET

# Step 2: Run evaluation
echo "Starting evaluation on $DATASET..."
python evaluate_hf_models.py \
  --verbose \
  --enabled_datasets $DATASET #\ 
  # --max_epochs 1 \
