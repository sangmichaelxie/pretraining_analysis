CACHE=cache
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
ROOT="."
LOGDIR=logs
mkdir -p $LOGDIR
