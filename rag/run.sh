#!/usr/bin/env bash
#SBATCH --mem=100000
#SBATCH --time=0
#SBATCH --gres=gpu:1

server=tir

mode=$1  # e2ec
hop=$2
model=$3  # facebook/rag-sequence-nq facebook/rag-sequence-base
source=$4  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project_noc.jsonl.source  nq/test.source
target=$5  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project_noc.jsonl.target  nq/test.target
out=$6  # hotpotqa/dev_select_project.pred

if [[ ${mode} == 'e2e' ]]; then
    ndocs=5
elif [[ ${mode} == 'break' ]]; then
    ndocs=5
elif [[ ${mode} == 'pseudo_break' ]]; then
    ndocs=1
elif [[ ${mode} == 'e2ec' ]]; then
    ndocs=1
elif [[ ${mode} == 'retrieval_all' ]]; then
    ndocs=100
fi

if [[ ${server} == 'tir' ]]; then
    prefix=""
elif [[ ${server} == 'nanhang' ]]; then
    gpu=$7
    prefix="CUDA_VISIBLE_DEVICES=${gpu} proxychains4"
fi

#CUDA_VISIBLE_DEVICES=${gpu} proxychains4
python eval_rag.py \
    --model_name_or_path ${model} \
    --model_type rag_sequence \
    --evaluation_set ${source} \
    --gold_data_path ${target} \
    --predictions_path ${out} \
    --eval_mode ${mode} \
    --retrieval_hop ${hop} \
    --gold_data_mode ans_tab \
    --num_beams 1 \
    --n_docs ${ndocs} \
    --eval_batch_size 128 \
    --print_predictions \
    --recalculate &> ${out}.out
