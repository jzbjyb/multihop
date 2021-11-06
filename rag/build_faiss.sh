#!/usr/bin/env bash
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0
#SBATCH --output=slurm_out/%j.out

server=$(hostname)

if [[ $server == 'GPU02' ]]; then
  export HF_DATASETS_CACHE='/mnt/data/zhengbao/.cache'
fi

input=$1
output=$2
rag_model=$3  # facebook/rag-sequence-nq
ctx_model=$4  # facebook/dpr-ctx_encoder-multiset-base
batch_size=256
m=32

python use_own_knowledge_dataset.py \
  --csv_path ${input} \
  --output_dir ${output} \
  --rag_model_name ${rag_model} \
  --dpr_ctx_encoder_model_name ${ctx_model} \
  --batch_size ${batch_size} \
  --m ${m}
