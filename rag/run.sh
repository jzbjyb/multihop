#!/usr/bin/env bash

model=facebook/rag-sequence-nq  # facebook/rag-sequence-base
out=nq/rag_empty.pred  # nq/raginit_empty.pred

CUDA_VISIBLE_DEVICES=0 proxychains4 python eval_rag.py \
    --model_name_or_path ${model} \
    --model_type rag_sequence \
    --evaluation_set nq/dev.source \
    --gold_data_path nq/dev.target \
    --predictions_path ${out} \
    --eval_mode e2e \
    --gold_data_mode ans \
    --num_beams 1 \
    --n_docs 1 \
    --eval_batch_size 128 \
    --print_predictions \
    --recalculate
