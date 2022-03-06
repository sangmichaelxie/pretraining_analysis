#!/bin/bash

source consts.sh

set -x

DATA_NAME=$1
DATA_DIR=$ROOT/data/$DATA_NAME

EPOCHS=${2:-3}
BATCH_SIZE=${3:-8}
OTHER_ARGS=$4

python src/run_mlm.py \
    --model_type bert \
    --tokenizer_name $DATA_DIR/tokenizer.json \
    --small_model \
    --custom_tokenizer \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --learning_rate 1e-5 \
    --num_train_epochs $EPOCHS \
    --warmup_steps 1000 \
    --output_dir $ROOT/output/${DATA_NAME}_pretrain \
    --logging_steps 100 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --save_steps 1500 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --fp16 \
    ${OTHER_ARGS}
