#!/usr/bin/env bash

mode=$1  # e2ec
model=$2  # facebook/rag-sequence-nq facebook/rag-sequence-base
source=$3  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project_noc.jsonl.source  nq_raw/val.source
target=$4  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project_noc.jsonl.target  nq_raw/val.target
out=$5  # hotpotqa/dev_select_project.pred
gpu=$6

if [[ ${mode} == 'e2e' ]]; then
    ndocs=5
elif [[ ${mode} == 'e2ec' ]]; then
    ndocs=1
fi

CUDA_VISIBLE_DEVICES=${gpu} proxychains4 python eval_rag.py \
    --model_name_or_path ${model} \
    --model_type rag_sequence \
    --evaluation_set ${source} \
    --gold_data_path ${target} \
    --predictions_path ${out} \
    --eval_mode ${mode} \
    --gold_data_mode ans_tab \
    --num_beams 1 \
    --n_docs ${ndocs} \
    --eval_batch_size 128 \
    --print_predictions \
    --recalculate
