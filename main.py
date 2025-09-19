

import argparse
import numpy as np
import torch

from enum import Enum
from pathlib import Path
from utils import *
from buffer import dataset_to_replaybuffer
from finitedice import FiniteOptiDICE
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2
from evaluate import evaluate_policy

# ---- import 환경 및 함수 ----
# from your_module import Fair_Taxi_MDP_Penalty_V2, dataset_to_replaybuffer, FiniteOptiDICE, FDivergence


# def get_config():

def main():
    parser = argparse.ArgumentParser()

    # ---- Experiment setup ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=100)

    # ---- Env setup ----
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--fuel", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="./outputs/")
    parser.add_argument("--dimension", type=int, default=2)

    # ---- OptiDICE setup ----
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128,128, 128])
    parser.add_argument("--time_embed_dim", type=int, default=8)
    parser.add_argument("--layer_norm", type=bool, default=True)

    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--nu_lr", type=float, default=3e-4)
    parser.add_argument("--f_divergence", type=str, default="Chi",
                        choices=[d.value for d in FDivergence])
    parser.add_argument("--nu_grad_penalty_coeff", type=float, default=0.001)

    # ---- Dataset / Replay buffer ----
    parser.add_argument("--dataset_path", type=str, default="./data/fair_taxi_dataset_v1.npy")
    parser.add_argument("--reward_index", type=int, default=0)
    parser.add_argument("--one_hot_pass_idx", type=bool, default=False)

    args = parser.parse_args()

    # reshape coords
    loc_coords = [[0,0],[3,2]]
    dest_coords = [[0,4],[3,3]]

    # return args


    # def main():
        # config = get_config()
    config = args
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    print(config)
    # ---- Env init ----
    size, loc_coords, dest_coords = get_setting(args.size, args.dimension)

    env = Fair_Taxi_MDP_Penalty_V2(
        size=size,
        loc_coords=loc_coords,
        dest_coords=dest_coords,
        fuel=config.fuel,
        output_path=config.output_path
    )

    # ---- Load dataset ----
    dataset = np.load(config.dataset_path, allow_pickle=True).item()

    utility = Utility(kind="log", weights=None, shift=0.0)

    # ---- Normalization ----
    keys = ['rewards']
    normalized_dataset, norm_stat_dict = normalize_dataset(dataset, keys, method="minmax")
    # Raccs 재계산
    normalized_dataset['Raccs'] = recompute_raccs(normalized_dataset, horizon=config.horizon)

    # ---- Convert to replay buffer ----
    buffer = dataset_to_replaybuffer(
        normalized_dataset,
        env,
        device=config.device,
        reward_index=config.reward_index,
        one_hot_pass_idx=config.one_hot_pass_idx,
        concat_acc_reward=True,
        reward_mode="esr",
        utility=utility
    )


    # # ---- Update config with state/action dims ----
    config.state_dim = env.decode(env.observation_space.n - 1).__len__()

    config.state_dim += 2
    config.action_dim = env.action_space.n

    # ---- Initialize OptiDICE ----
    config.f_divergence = FDivergence(config.f_divergence)
    agent = FiniteOptiDICE(config, device=config.device)

    print("✅ Environment and agent initialized.")
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}")
    print(f"Dataset size: {len(dataset['states'])}")



    def train(agent, buffer, num_steps=10000, batch_size=256, log_interval=100):
        stats_history = []

        for step in range(1, num_steps + 1):
            batch = buffer.sample(batch_size)
            stats = agent.train_step(batch)
            
            if step % log_interval == 0:
                print(f"[Step {step}] " + ", ".join([f"{k}: {v:.4f}" for k,v in stats.items()]))
            stats_history.append(stats)
        
        return stats_history


    stats_history = train(
        agent,
        buffer,
        num_steps= config.num_steps,
        batch_size= config.batch_size,
        log_interval=config.log_interval
    )
    
    eval_results = evaluate_policy(
        env,
        agent,
        num_episodes=20,
        config=config,
        max_steps=config.horizon,
        render=False,
        normalize=True,                # 학습 때 normalization 했다면 True
        norm_stats=norm_stat_dict,     # normalize_dataset 리턴값
        utility=utility                # ESR 학습에 썼던 Utility
    )

    print("📊 Evaluation Results")
    print("Average scalar return:", eval_results["avg_scalar_return"])
    print("Average vector return:", eval_results["avg_vector_return"])

    
if __name__ == "__main__":
    main()
