#!/usr/bin/env bash
#SBATCH --mem=100000
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0
#SBATCH --output=slurm_out/%j.out

server=tir

mode=$1  # e2ec
hop=$2
model=$3  # facebook/rag-sequence-nq facebook/rag-sequence-base
index_path=$4
if [[ ${index_path} == 'none' ]]; then
    index_path=''
else
    index_path="--index_path ${index_path}"
fi
source=$5  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project_noc.jsonl.source  nq/test.source
target=$6  # ../../Break/break_dataset/QDMR-high-level/hotpotqa/dev_select_project_noc.jsonl.target  nq/test.target
out=$7  # hotpotqa/dev_select_project.pred
num_beams=1
batch_size=128
gmethod=generator

add=''
if [[ ${mode} == 'e2e' ]]; then
    ndocs=5
    max_source_length=128
elif [[ ${mode} == 'e2e_multichoice' ]]; then
    ndocs=5
    max_source_length=128
elif [[ ${mode} == 'break' ]]; then
    ndocs=5
    max_source_length=128
elif [[ ${mode} == 'pseudo_break' ]]; then
    ndocs=1
    max_source_length=512
elif [[ ${mode} == 'e2ec' ]]; then
    ndocs=1
    max_source_length=512
elif [[ ${mode} == 'e2ec_multichoice' ]]; then
    ndocs=1
    max_source_length=512
elif [[ ${mode} == 'e2ec_nq' ]]; then
    ndocs=1
    max_source_length=512
    add='--no_question'
elif [[ ${mode} == 'retrieval_all' ]]; then
    ndocs=100
    max_source_length=128
fi

if [[ ${server} == 'tir' ]]; then
    prefix=""
elif [[ ${server} == 'nanhang' ]]; then
    gpu=$8
    export CUDA_VISIBLE_DEVICES=${gpu}
fi

python eval_rag.py \
    --model_name_or_path ${model} \
    --model_type rag_sequence \
    --evaluation_set ${source} \
    --gold_data_path ${target} \
    --predictions_path ${out} \
    --eval_mode ${mode} \
    --retrieval_hop ${hop} \
    --gold_data_mode ans_tab \
    --num_beams ${num_beams} \
    --num_return_sequences ${num_beams} \
    --n_docs ${ndocs} \
    --max_source_length ${max_source_length} \
    --eval_batch_size ${batch_size} \
    --print_predictions \
    --generation_method ${gmethod} \
    --recalculate ${add} ${index_path}
