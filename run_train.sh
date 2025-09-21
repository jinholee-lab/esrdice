#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 5개의 random seed 값 목록
SEEDS=(1 2 3 4 5)

# alpha sweep 값 목록
ALPHAS=(10 1 0.1 0.01 0.001)

for seed in "${SEEDS[@]}"; do
    echo "Running experiments for seed=${seed}"

    for alpha in "${ALPHAS[@]}"; do
    echo "Running experiment with alpha=${alpha}"

    python main.py \
        --seed ${seed} \
        --device "cuda" \
        --num_steps 50000 \
        --batch_size 256 \
        --log_interval 1000 \
        --size 10 \
        --fuel 100 \
        --reward_dim 2 \
        --utility_kind "piecewise_log" \
        --horizon 100 \
        --alpha ${alpha} \
        --hidden_dims 128 128 \
        --time_embed_dim 8 \
        --layer_norm True \
        --policy_lr 3e-4 \
        --nu_lr 3e-4 \
        --f_divergence "Chi" \
        --nu_grad_penalty_coeff 0.001 \
        --save_path "./checkpoints/" \
        --dataset_path "./data/dataset_v3.npy" \
        --one_hot_pass_idx True \
        --concat_acc_reward True \
        --normalization_method "linear" \
        --use_wandb True \
        --tag "ESRDICE" &

    done

    wait
    echo "Completed all experiments for seed=${seed}"
done
