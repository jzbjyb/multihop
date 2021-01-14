#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 proxychains4 python eval_rag.py \
    --model_name_or_path facebook/rag-sequence-nq \
    --model_type rag_sequence \
    --evaluation_set nq/dev.source \
    --gold_data_path nq/dev.target \
    --predictions_path nq/dev.pred \
    --eval_mode e2e \
    --gold_data_mode ans \
    --num_beams 1 \
    --n_docs 1 \
    --eval_batch_size 128 \
    --print_predictions \
    --recalculate
