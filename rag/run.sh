#!/usr/bin/env bash

model=facebook/rag-sequence-nq  # facebook/rag-sequence-base
out=hotpotqa/rag.pred  # nq/raginit_empty.pred
mode=e2ec
#source=nq/dev.source
#target=nq/dev.target
source=../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project.jsonl.source
target=../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project.jsonl.target

CUDA_VISIBLE_DEVICES=0 proxychains4 python eval_rag.py \
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
