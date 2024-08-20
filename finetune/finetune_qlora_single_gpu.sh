#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

MODEL="output_qwen_qg" # Qwen/Qwen-VL-Chat-Int4 Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="data/ScienceQA_train_rg.json"

export CUDA_VISIBLE_DEVICES=0

# Remember to use --fp16 instead of --bf16 due to autogptq
CUDA_VISIBLE_DEVICES=0 python distractor_network.py \
    --local_only True \
    --use_checkpoint True \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 False \
    --fix_vit True \
    --output_dir output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --use_lora \
    --q_lora \
    --deepspeed finetune/ds_config_zero2.json
