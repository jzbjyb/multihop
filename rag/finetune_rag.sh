#!/usr/bin/env bash
#SBATCH --mem=100000
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0
#SBATCH --output=slurm_out/%j.out

# wandb
export WANDB_PROJECT="adapt-knowledge"

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

data=$1  # /home/jzb/exp/Break/break_dataset/QDMR-high-level/hotpotqa/full_qa_small
model1=$2 # facebook/rag-sequence-nq
model2=models_more/cwq_goldneg/rag_ssm_explicit/checkpoint11
index_path=$3
max_combined_length=256
max_source_length=128
max_target_length=128
n_docs=1
mode=ret
consistency_loss=no
distance=jsd
multitask=no
hop=1
output=$4  # models/rag_combine
gpus=$5
IFS=',' read -ra gpus_list <<< "$gpus"  # split gpus by ','
ngpus=${#gpus_list[@]}  # count the number of gpus
port=$6
batch_size=4
epochs=3
freeze=none
negative_method=none
memory_bank_size=500

#HF_HOME=$HOME/tir4/hf_home_cache
CUDA_VISIBLE_DEVICES=${gpus} python finetune_rag.py \
    --logger_name default \
    --data_dir ${data} \
    --output_dir ${output} \
    --model_name_or_path ${model1} \
    --model_type rag_sequence \
    --index_path ${index_path} \
    --fp16 \
    --gpus ${ngpus} \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${batch_size} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --val_max_target_length ${max_target_length} \
    --test_max_target_length ${max_target_length} \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs ${epochs} \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --retrieval_mode ${mode} \
    --retrieval_hop ${hop} \
    --max_combined_length ${max_combined_length} \
    --distributed-port ${port} \
    --consistency_loss ${consistency_loss} \
    --distance ${distance} \
    --multitask ${multitask} \
    --freeze ${freeze} \
    --negative_method ${negative_method} \
    --memory_bank_size ${memory_bank_size}
