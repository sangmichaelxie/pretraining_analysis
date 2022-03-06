# Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning

The codebase generates a synthetic pretraining dataset from an HMM. The downstream task is a linear function of the posterior distribution of the first HMM hidden state in the sequence. We provide code for two experiments:
- Head tuning vs. prompt tuning: We generate data from a vanilla HMM and train a Transformer masked language model (MLM). We compare the downstream accuracy of head tuning (linear probing on the CLS token) vs. prompt tuning (head tuning + optimized continuous input vectors prepended to the input). Run all seeds and hidden state sizes using `scripts/exp1_local_runner.sh` or `scripts/exp1_slurm_runner.sh` on Slurm.
- Memory-augmented HMMs: We consider generating data from a memory-augmented HMM, where the memory structure allows for relaxed conditions for learning useful representations for the downstream task. Run all seeds and memory slot sizes/hidden state sizes using `scripts/exp2_local_runner.sh` or `scripts/exp2_slurm_runner.sh` on Slurm.

## Setup
To get started, run the following:
```
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
pip install -e transformers
cd apex
pip install --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Please see the [paper](https://arxiv.org/abs/2106.09226) for more details.
```
@article{wei2021why,
         author = {Colin Wei and Sang Michael Xie and Tengyu Ma},
         journal = {Neural Information Processing Systems (NeurIPS)},
         title = {Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning},
         year = {2021},}
```

