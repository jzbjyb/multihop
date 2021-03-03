# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options

DATA_DIR=nq_small  # /home/jzb/exp/Break/break_dataset/QDMR-high-level/hotpotqa/full_qa_small
MODEL_NAME_OR_PATH=facebook/rag-sequence-nq  # facebook/rag-sequence-nq
max_combined_length=200
mode=$1
hop=$2
OUTPUT_DIR=$3  # models/rag_combine
gpus=$4
ngpus=$5
port=$6
batch_size=4  # $(( 1*${ngpus} ))

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
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-06 \
    --num_train_epochs 2 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --retrieval_mode ${mode} \
    --retrieval_hop ${hop} \
    --max_combined_length ${max_combined_length} \
    --distributed-port ${port} \
    --use_mdr
