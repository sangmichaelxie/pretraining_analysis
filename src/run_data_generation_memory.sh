#!/bin/bash

source consts.sh

TRANS_TEMP=$1
EMISSION_TEMP=$2
START_TEMP=$3
N_SYMBOLS=$4
N_HIDDEN_STATES=$5
N_MEM_SLOTS=$6
N_EXAMPLES=$7
OTHER_ARGS=$8
python src/generate_data_memory.py \
    --transition_temp $TRANS_TEMP \
    --emission_temp $EMISSION_TEMP \
    --start_temp $START_TEMP \
    --n_symbols $N_SYMBOLS \
    --n_memory_slots $N_MEM_SLOTS \
    --n_hidden_states $N_HIDDEN_STATES \
    --n_examples $N_EXAMPLES \
    --data_dir $ROOT/data \
    $OTHER_ARGS

