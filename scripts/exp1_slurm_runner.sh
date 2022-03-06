#!/bin/bash

source consts.sh

MLM_PROB=0.05
PREFIX_LEN=20
EMISSION_TEMP=0.01
N_SYMBOLS=10
TRANS_TEMP=0.01
for N_COMPONENTS in 4 8 10 12 15 25 30
do
for SEED in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
sbatch \
    --output $LOGDIR/trans${TRANS_TEMP}_emit${EMISSION_TEMP}_nsym${N_SYMBOLS}_NCOM${N_COMPONENTS}_nexamples5000_seed${SEED} \
    --mem 16g \
    --gres=gpu:1 \
    scripts/run_exps_seed.sh \
        $TRANS_TEMP \
        $EMISSION_TEMP \
        $N_SYMBOLS \
        $N_COMPONENTS \
        $SEED \
        "--mlm_probability ${MLM_PROB}" \
        "--prefix_len ${PREFIX_LEN}" \
        "--n_examples 5000 --seed ${SEED}"
done
done


