export WANDB_DISABLED=true

# dataset=chess100k_gold_s_asa_n4
dataset=chess10k_gold_s_asa
# dataset=chess_test

# Create output directory with name reflecting configuration
exp=output/$dataset/DDM-s_asa-bs64-ga2-lr3e-4-ep300-T20-$(date "+%Y%m%d-%H%M%S")
mkdir -p "$exp/${dataset}"

# Check GPU count first, before we use it
ngpu=$(nvidia-smi --list-gpus | wc -l)

# Use a variable for the checkpoint path for easier modification
checkpoint_path="output/chess10k_gold_s_asa/DDM-s_asa-bs1024-lr3e-4-ep200-T20-20250428-171543"

echo ">>> Using checkpoint: $checkpoint_path"
if [ -n "$checkpoint_path" ] && [ -d "$checkpoint_path" ]; then
    echo ">>> Resuming training from checkpoint: $checkpoint_path"
    if [ "$ngpu" -ge 2 ]; then
        echo ">>> MultiGPU Running"
        CUDA_VISIBLE_DEVICES=0,1 \
        accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
        src/train_bash.py \
            --stage ddm --overwrite_output_dir \
            --cache_dir ./cache \
            --model_name_or_path chess_config \
            --do_train \
            --dataset $dataset \
            --finetuning_type full \
            --cutoff_len 328 \
            --output_dir $exp \
            --overwrite_cache \
            --per_device_train_batch_size 64 \
            --gradient_accumulation_steps 2 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --val_size 448 \
            --per_device_eval_batch_size 32 \
            --eval_steps 50 \
            --evaluation_strategy steps \
            --save_steps 200 \
            --learning_rate 3e-4 \
            --learning_rate 5e-6 \
            --warmup_ratio 0.05 \
            --num_train_epochs 300.0 \
            --plot_loss \
            --run_name ${dataset}_prefix \
            --preprocessing_num_workers 8 \
            --fp16 \
            --save_total_limit 3 \
            --remove_unused_columns False \
            --diffusion_steps 20 \
            --save_safetensors False \
            --time_reweighting cosine \
            --topk_decoding True \
            --resume_from_checkpoint $checkpoint_path \
            > $exp/train.log
    else 
        echo ">>> Single GPU Training"
        # Single GPU Training 
        CUDA_VISIBLE_DEVICES=0 \
        RANK=0 \
        WORLD_SIZE=1 \
        accelerate launch --num_machines 1 --mixed_precision fp16 --num_processes 1 --main_process_port 20099 \
        src/train_bash.py \
            --stage ddm --overwrite_output_dir \
            --cache_dir ./cache \
            --model_name_or_path chess_config \
            --do_train \
            --dataset $dataset \
            --finetuning_type full \
            --cutoff_len 328 \
            --output_dir $exp \
            --overwrite_cache \
            --per_device_train_batch_size 64 \
            --gradient_accumulation_steps 2 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --val_size 448 \
            --per_device_eval_batch_size 32 \
            --eval_steps 50 \
            --evaluation_strategy steps \
            --save_steps 200 \
            --learning_rate 3e-4 \
            --learning_rate 5e-6 \
            --warmup_ratio 0.05 \
            --num_train_epochs 300.0 \
            --plot_loss \
            --run_name ${dataset}_prefix \
            --preprocessing_num_workers 8 \
            --fp16 \
            --save_total_limit 3 \
            --remove_unused_columns False \
            --diffusion_steps 20 \
            --save_safetensors False \
            --time_reweighting cosine \
            --topk_decoding True \
            --resume_from_checkpoint $checkpoint_path \
            > $exp/train.log
    fi

else 
    echo ">>> No valid checkpoint found at $checkpoint_path. Starting training from scratch..."
    if [ "$ngpu" -ge 2 ]; then
        echo ">>> MultiGPU Running"
        # MultiGPU Training Process
        CUDA_VISIBLE_DEVICES=0,1 \
        accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
        src/train_bash.py \
            --stage ddm --overwrite_output_dir \
            --cache_dir ./cache \
            --model_name_or_path chess_config \
            --do_train \
            --dataset $dataset \
            --finetuning_type full \
            --cutoff_len 328 \
            --output_dir $exp \
            --overwrite_cache \
            --per_device_train_batch_size 64 \
            --gradient_accumulation_steps 2 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --val_size 448 \
            --per_device_eval_batch_size 32 \
            --eval_steps 50 \
            --evaluation_strategy steps \
            --save_steps 200 \
            --learning_rate 3e-4 \
            --learning_rate 5e-6 \
            --warmup_ratio 0.05 \
            --num_train_epochs 300.0 \
            --plot_loss \
            --run_name ${dataset}_prefix \
            --preprocessing_num_workers 8 \
            --fp16 \
            --save_total_limit 3 \
            --remove_unused_columns False \
            --diffusion_steps 20 \
            --save_safetensors False \
            --time_reweighting cosine \
            --topk_decoding True \
            > $exp/train.log
    else
        echo ">>> Single GPU Training"
        # Single GPU Training 
        CUDA_VISIBLE_DEVICES=0 \
        RANK=0 \
        WORLD_SIZE=1 \
        accelerate launch --num_machines 1 --mixed_precision fp16 --num_processes 1 --main_process_port 20099 \
        src/train_bash.py \
            --stage ddm --overwrite_output_dir \
            --cache_dir ./cache \
            --model_name_or_path chess_config \
            --do_train \
            --dataset $dataset \
            --finetuning_type full \
            --cutoff_len 328 \
            --output_dir $exp \
            --overwrite_cache \
            --per_device_train_batch_size 64 \
            --gradient_accumulation_steps 2 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --val_size 448 \
            --per_device_eval_batch_size 32 \
            --eval_steps 50 \
            --evaluation_strategy steps \
            --save_steps 200 \
            --learning_rate 3e-4 \
            --learning_rate 5e-6 \
            --warmup_ratio 0.05 \
            --num_train_epochs 300.0 \
            --plot_loss \
            --run_name ${dataset}_prefix \
            --preprocessing_num_workers 8 \
            --fp16 \
            --save_total_limit 3 \
            --remove_unused_columns False \
            --diffusion_steps 20 \
            --save_safetensors False \
            --time_reweighting cosine \
            --topk_decoding True \
            > $exp/train.log
    fi
fi

# Run evaluation
echo ">>> Running evaluation..."
CUDA_VISIBLE_DEVICES=0 \
python3 -u src/train_bash.py \
    --stage ddm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path chess_config \
    --do_predict \
    --cutoff_len 328 \
    --dataset chess_test \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-cosine \
    --topk_decoding True \
    > $exp/${dataset}/eval.log

echo ">>> Training and evaluation completed. Results in $exp"