#!/bin/bash

source consts.sh

MLM_PROB=0.05
PREFIX_LEN=20
N_HIDDEN_STATES=4
TRANS_TEMP=0.01
EMISSION_TEMP=0.01
N_SYMBOLS=10

for N_MEM_SLOTS in 2 3 5 7
do
for SEED in 1 2 3 4 5
do
LOGFILE=$LOGDIR/VANILLA_trans${TRANS_TEMP}_emit${EMISSION_TEMP}_nsym${N_SYMBOLS}_nexamples5000_nhid${N_HIDDEN_STATES}_nmem${N_MEM_SLOTS}_seed${SEED}
bash scripts/run_exps_memory_vanilla_seed.sh \
        $TRANS_TEMP \
        $EMISSION_TEMP \
        $N_SYMBOLS \
        $N_HIDDEN_STATES \
        $N_MEM_SLOTS \
        $SEED \
        "--mlm_probability ${MLM_PROB}" \
        "--prefix_len ${PREFIX_LEN}" \
        "--vanilla --seed ${SEED}" 2>&1 | tee $LOGFILE
done
done

for N_MEM_SLOTS in 2 3 5 7
do
for SEED in 1 2 3 4 5
do
LOGFILE=$LOGDIR/MEMORY_trans${TRANS_TEMP}_emit${EMISSION_TEMP}_nsym${N_SYMBOLS}_nexamples5000_nhid${N_HIDDEN_STATES}_nmem${N_MEM_SLOTS}_seed${SEED}
bash scripts/run_exps_memory_seed.sh \
        $TRANS_TEMP \
        $EMISSION_TEMP \
        $N_SYMBOLS \
        $N_HIDDEN_STATES \
        $N_MEM_SLOTS \
        $SEED \
        "--mlm_probability ${MLM_PROB}" \
        "--prefix_len ${PREFIX_LEN}" \
        "--seed ${SEED}" 2>&1 | tee $LOGFILE
done
done
