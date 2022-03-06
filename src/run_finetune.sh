#!/bin/bash

source consts.sh

set -x

DATA_NAME=$1
DATA_DIR=$ROOT/data/$DATA_NAME
TRAIN_MODE=$2
LR=$3
OTHER_ARGS=$4

python src/run_finetune.py \
    --max_seq_length 80 \
    --train_file $DATA_DIR/downstream_train.json \
    --validation_file $DATA_DIR/downstream_val.json \
    --test_file $DATA_DIR/downstream_test.json \
    --tokenizer_name $DATA_DIR/tokenizer.json \
    --training_mode ${TRAIN_MODE} \
    --do_eval \
    --do_predict \
    --learning_rate ${LR} \
    --num_train_epochs 100 \
    --warmup_steps 500 \
    --output_dir $ROOT/output/${DATA_NAME}_${TRAIN_MODE} \
    --do_train \
    --overwrite_output_dir \
    --model_name_or_path $ROOT/output/${DATA_NAME}_pretrain \
    --logging_steps 100 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --save_steps 1000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --custom_tokenizer \
    --prepend_cls \
    --remove_unused_columns False \
    $OTHER_ARGS
