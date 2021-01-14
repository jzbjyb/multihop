#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python eval_rag.py \
    --model_name_or_path facebook/rag-sequence-nq \
    --model_type rag_sequence \
    --evaluation_set nq/dev.source \
    --gold_data_path nq/dev.target \
    --predictions_path nq/dev.pred \
    --eval_mode e2e \
    --gold_data_mode qa \
    --n_docs 0 \
    --print_predictions \
    --recalculate
