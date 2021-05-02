#!/usr/bin/env bash
#SBATCH --mem=100000
#SBATCH --time=0
#SBATCH --gres=gpu:v100:1

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options

DATA_DIR=$1  # /home/jzb/exp/Break/break_dataset/QDMR-high-level/hotpotqa/full_qa_small
MODEL_NAME_OR_PATH=facebook/rag-sequence-nq # facebook/rag-sequence-nq
MODEL2=models_more/cwq_goldneg/rag_ssm_explicit/checkpoint11
max_combined_length=512
max_target_length=128
n_docs=1
mode=no
consistency_loss=$2
distance=$3
multitask=no
hop=1
OUTPUT_DIR=$4  # models/rag_combine
gpus=$5
ngpus=$6
port=$7
batch_size=4  # $(( 1*${ngpus} ))

#CUDA_VISIBLE_DEVICES=${gpus} proxychains4
#HF_HOME=$HOME/tir4/hf_home_cache
CUDA_VISIBLE_DEVICES=${gpus} proxychains4 python finetune_rag.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type rag_sequence \
    --fp16 \
    --gpus ${ngpus} \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --train_batch_size ${batch_size} \
    --eval_batch_size 1 \
    --max_source_length ${max_combined_length} \
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
    --num_train_epochs 5 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --retrieval_mode ${mode} \
    --retrieval_hop ${hop} \
    --max_combined_length ${max_combined_length} \
    --distributed-port ${port} \
    --consistency_loss ${consistency_loss} \
    --distance ${distance} \
    --multitask ${multitask}
