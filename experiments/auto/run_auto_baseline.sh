#!/usr/bin/env bash

python experiments/auto/run_auto_classifier.py \
  --task_name TEMPORAL \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir dataset \
  --model_name $1 \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --cuda 0

python evaluator/auto_evaluator.py eval --model_name $1