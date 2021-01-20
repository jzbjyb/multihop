#!/usr/bin/env bash

mode=e2ec
model=facebook/rag-sequence-nq  # facebook/rag-sequence-base
#source=nq/dev.source
#target=nq/dev.target
source=$1  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project.jsonl.source
target=$2  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project.jsonl.target
out=$3  # hotpotqa/dev_select_project.pred
gpu=$4

CUDA_VISIBLE_DEVICES=${gpu} proxychains4 python eval_rag.py \
    --model_name_or_path ${model} \
    --model_type rag_sequence \
    --evaluation_set ${source} \
    --gold_data_path ${target} \
    --predictions_path ${out} \
    --eval_mode ${mode} \
    --gold_data_mode ans \
    --num_beams 1 \
    --n_docs 1 \
    --eval_batch_size 128 \
    --print_predictions \
    --recalculate
