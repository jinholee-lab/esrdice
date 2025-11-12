

import argparse
import json
import os
import numpy as np
import torch

from enum import Enum
from pathlib import Path
from utils import get_setting
from buffer import ReplayBuffer
from finitedice import FiniteOptiDICE
from finitefairdice import FiniteFairDICE
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2
from evaluate import evaluate_policy
from divergence import FDivergence
from utility import Utility
import wandb

# ---- import 환경 및 함수 ----
# from your_module import Fair_Taxi_MDP_Penalty_V2, dataset_to_replaybuffer, FiniteOptiDICE, FDivergence

import argparse 
import numpy as np

def preprocess_timesteps(dataset):
    timesteps = dataset["timesteps"].astype(np.int64)  # [N]
    next_timesteps = timesteps + 1

    new_dataset = dataset.copy()
    new_dataset["next_timesteps"] = next_timesteps
    return new_dataset


def preprocess_rewards(dataset, method="linear"):
    """
    dataset: dict, includes 'rewards' key with shape [N, D]
    method: str, "zscore" | "minmax" | "linear"
    """
    new_dataset = dataset.copy()
    if "rewards" not in dataset:
        raise KeyError(f"rewards is not in dataset")

    arr = dataset["rewards"]
    stat_dict = {}
    
    if method == "linear":
        # get max value from entire dataset, since the env reward design is same for all dimensions
        max_val = arr.max() 
        if max_val > 0:
            new_dataset["rewards"] = arr / max_val
        else:
            raise ValueError("max reward is non-positive, please check the dataset")
        stat_dict["rewards"] = {"max" : max_val}
    elif method == "zscore":
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            new_dataset["rewards"] = (arr - mean) / std
        else:
            new_dataset["rewards"] = arr 
        stat_dict["rewards"] = {"mean" : mean,
                                "std" : std}
    elif method == "minmax":
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            new_dataset["rewards"] = (arr - min_val) / (max_val - min_val)
        else:
            new_dataset["rewards"] = arr  # 값이 모두 같으면 그대로 둠
        stat_dict["rewards"] = {"max" : max_val,
                                "min" : min_val}
    else:
        # no normalization
        new_dataset["rewards"] = arr
        max_val = arr.max() 
        stat_dict["rewards"] = {"max" : max_val}

    return new_dataset, stat_dict


def preprocess_Raccs(dataset, horizon):
    """
    rewards를 누적해 Raccs, next_Raccs를 만들어 dataset에 추가
    
    Args:
      dataset: dict, 반드시 'rewards' [N, D], 'timesteps' [N] 포함
      horizon: 에피소드 길이 (fixed horizon)
      
    Returns:
      new_dataset: 원본 dataset + 'Raccs', 'next_Raccs' 추가
    """
    rewards = dataset["rewards"]  # [N, D]
    N, D = rewards.shape

    Raccs = np.zeros((N, D), dtype=np.float32)
    next_Raccs = np.zeros((N, D), dtype=np.float32)

    # trajectory 단위로 처리 (timesteps가 0~horizon-1로 반복된다고 가정)
    for start in range(0, N, horizon):
        acc = np.zeros(D, dtype=np.float32)
        for t in range(horizon):
            idx = start + t
            Raccs[idx] = acc                  # R_acc,t
            acc = acc + rewards[idx]          # update
            next_Raccs[idx] = acc             # R_acc,t+1

    new_dataset = dataset.copy()
    new_dataset["Raccs"] = Raccs
    new_dataset["next_Raccs"] = next_Raccs
    return new_dataset

def preprocess_states(
    dataset,
    env,
    one_hot_xy=False,
    one_hot_pass_idx=True,
    concat_raccs=True,
):
    """
    dataset: dict, keys 포함
      - 'states', 'next_states', 'initial_states' : [N] int codes
      - 'Raccs' : [N, D] (optional, concat_raccs=True일 때 필요)
    env: Taxi environment (decode() 지원)
    
    옵션:
      one_hot_xy   : taxi (x,y)를 one-hot 인코딩할지 여부
      one_hot_pass : passenger 상태를 one-hot 인코딩할지 여부
      concat_raccs : Raccs를 feature에 붙일지 여부
    
    return: new_dataset with processed states
    """

    def decode_and_feat(states, raccs=None):
        decoded = np.array([env.decode(s) for s in states])  # (N,4)
        taxi_x, taxi_y, pass_loc, pass_idx = decoded.T

        feats = []

        # --- Taxi 위치 ---
        if one_hot_xy:
            taxi_x_oh = np.eye(env.size)[taxi_x]
            taxi_y_oh = np.eye(env.size)[taxi_y]
            feats.append(taxi_x_oh)
            feats.append(taxi_y_oh)
        else:
            feats.append(taxi_x[:, None])
            feats.append(taxi_y[:, None])

        # --- Passenger 상태 ---
        
        # pass_loc: binary (0: no one in taxi, 1: in taxi)
        feats.append(pass_loc[:, None])

        if one_hot_pass_idx:
            pass_idx_oh = np.eye(len(env.dest_coords) + 1)[pass_idx]
            feats.append(pass_idx_oh)
        else:
            feats.append(pass_idx[:, None])

        feats = np.concatenate(feats, axis=1)

        # --- Raccs 붙이기 ---
        if concat_raccs and raccs is not None:
            feats = np.concatenate([feats, raccs], axis=1)

        return feats

    # ---- dataset의 세 가지 state 전처리 ----
    new_dataset = dataset.copy()
    new_dataset['states'] = decode_and_feat(dataset["states"], dataset.get("Raccs"))
    new_dataset['next_states'] = decode_and_feat(dataset["next_states"], dataset.get("next_Raccs"))
    new_dataset['initial_states'] = decode_and_feat(dataset["initial_states"], np.zeros_like(dataset['Raccs']))  # init은 raccs 0으로 채워도 무방

    return new_dataset

def preprocess_scalarization(dataset, utility, keep_dims=False, horizon=100, add_u0_to="first"):
    """
    ESR step-wise 보상을 계산하고, 에피소드당 한 번 u(0)를 더해
    총합이 정확히 u(R_T)와 일치하도록 보정합니다.
    """
    Raccs   = dataset["Raccs"]      # [N, D]
    rewards = dataset["rewards"]    # [N, D]
    N, D    = rewards.shape

    # 1) ESR: u(R_{t+1}) - u(R_t)
    esr = utility(Raccs + rewards, keep_dims=keep_dims) - utility(Raccs, keep_dims=keep_dims)
 
    # 2) 에피소드 보정: 한 번만 u(0) 더하기 (합 == u(R_T))
    if horizon is not None:
        if N % horizon != 0:
            raise ValueError(f"N={N} is not a multiple of horizon={horizon}. "
                             "Use timesteps-based correction instead.")

        # u(0) 계산 (keep_dims에 따라 스칼라 또는 (D,))
        if keep_dims:
            u0 = utility(np.zeros((1, D), dtype=np.float32), keep_dims=True)[0]   # shape (D,)
        else:
            u0 = utility(np.zeros((1, D), dtype=np.float32), keep_dims=False)[0]  # scalar

        if add_u0_to == "first":
            # 각 에피소드의 첫 스텝 인덱스: 0, H, 2H, ...
            idx = np.arange(0, N, horizon)
        elif add_u0_to == "last":
            # 각 에피소드의 마지막 스텝 인덱스: H-1, 2H-1, ...
            idx = np.arange(horizon-1, N, horizon)
        else:
            raise ValueError("add_u0_to must be 'first' or 'last'.")

        esr[idx] += u0  # 브로드캐스팅으로 (N,) 또는 (N,D) 모두 안전

    # 3) dataset 업데이트
    new_dataset = dict(dataset)
    new_dataset["rewards"] = esr.astype(np.float32, copy=False)
    return new_dataset


def main():
    parser = argparse.ArgumentParser()

    # ---- Experiment setup ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)

    # ---- Env setup ----
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--fuel", type=int, default=100)
    parser.add_argument("--reward_dim", type=int, default=2)
    parser.add_argument("--utility_kind", type=str, default="piecewise_log")
    parser.add_argument("--output_path", type=str, default="./outputs/")

    # ---- Hyperparameter setup ----
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--time_embed_dim", type=int, default=8)
    parser.add_argument("--layer_norm", type=bool, default=True)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--nu_lr", type=float, default=3e-4)
    parser.add_argument("--mu_lr", type=float, default=3e-4)
    parser.add_argument("--f_divergence", type=str, default="Chi",
                        choices=[d.value for d in FDivergence])
    parser.add_argument("--nu_grad_penalty_coeff", type=float, default=0.001)
    parser.add_argument("--fair_alpha", type=float, default=1.0)
    parser.add_argument("--f_divergence_threshold", type=float, default=None)
    parser.add_argument("--lambda_lr", type=float, default=3e-4)
    parser.add_argument("--lambda_init", type=float, default=1e-5)
    parser.add_argument("--use_fixed_lambda", type=bool, default=False)
    
    # ---- path ----
    parser.add_argument("--save_path", type=str, default="./checkpoints/")

    # ---- Dataset / Replay buffer ----
    parser.add_argument("--dataset_path", type=str, default="./data/dataset_v3.npy")
    parser.add_argument("--one_hot_xy", type=bool, default=False)
    parser.add_argument("--one_hot_pass_idx", type=bool, default=False)
    parser.add_argument("--concat_acc_reward", type=bool, default=False)
    parser.add_argument("--normalization_method", type=str, default="linear", choices=["zscore", "minmax", "linear", "none"])

    # ---- logging ----
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="AET_v1")
    parser.add_argument("--tag", type=str, default="test")
    
    # ---- mode ----
    parser.add_argument("--mode", type=str, default="ser", choices=["esr", "ser","AET", "AET_constraint"])
    parser.add_argument("--policy_rollout", type=str, default="deterministic", choices=["stochastic", "deterministic"])

    args = parser.parse_args()

    config = args
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if config.mode == "AET" or config.mode == "AET_constraint":
        keep_dims = True
    else:
        keep_dims = False
        
    # ---- Env init ----
    config.size, config.loc_coords, config.dest_coords = get_setting(args.size, args.reward_dim)

    env = Fair_Taxi_MDP_Penalty_V2(
        size=config.size,
        loc_coords=config.loc_coords,
        dest_coords=config.dest_coords,
        fuel=config.fuel,
        output_path=config.output_path
    )

    # ---- Load dataset ----
    dataset = np.load(config.dataset_path, allow_pickle=True).item()
    
    # ---- Preprocess timesteps ----
    dataset = preprocess_timesteps(dataset)

    # ---- Preprocess rewards ----
    dataset, norm_stats = preprocess_rewards(dataset, method=config.normalization_method)
    
    # ---- Preprocess Raccs ----
    dataset = preprocess_Raccs(dataset, horizon=config.horizon)
    
    # ---- Preprocess scalarization ----
    utility = Utility(
        kind=config.utility_kind,
        alpha=1 - config.fair_alpha
        )

    # dataset = preprocess_scalarization(dataset, utility, keep_dims=keep_dims) # We do this in the buffer now
    
    # ---- Preprocess states ----
    dataset = preprocess_states(
        dataset,
        env,
        one_hot_xy=config.one_hot_xy,
        one_hot_pass_idx=config.one_hot_pass_idx,
        concat_raccs=config.concat_acc_reward, # Raccs
    )
    
    config.state_dim = dataset["states"].shape[1] + (config.reward_dim if not config.concat_acc_reward else 0)
    config.action_dim = env.action_space.n
    # if config.mode == "esr":
    #     dataset['rewards'] = dataset['scalarized_rewards']  # ESR 학습을 위해 scalarized reward로 교체
    config.reward_dim = dataset["rewards"].shape[1]  # reward dim (D)
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}, Reward dim: {config.reward_dim}")
    
    # ---- Replay Buffer ----
    buffer = ReplayBuffer(
        device=config.device,
        utility=utility,
        horizon=config.horizon,
        keep_dims=keep_dims,
        reward_dim=config.reward_dim
    )
    buffer.load_dataset(dataset)
    
    # ---- Initialize DICE ----
    config.f_divergence = FDivergence(config.f_divergence)

    if config.mode == "esr":
        agent = FiniteOptiDICE(config, device=config.device)
        print("Using FiniteOptiDICE for ESR optimization.")
    elif config.mode == "ser":
        agent = FiniteFairDICE(config, device=config.device)
        print("Using FiniteFairDICE for SER optimization.")
    elif config.mode == "AET":
        from AET import FiniteAET
        agent = FiniteAET(config, device=config.device)
    elif config.mode == "AET_constraint":
        from AET_constraint import FiniteAET
        agent = FiniteAET(config, device=config.device)
        print("Using FiniteAET for AET_constraint optimization.")
    else:
        raise ValueError("Invalid mode. Choose 'esr' or 'ser'.")
    
    print("✅ Environment and agent initialized.")
    print("Config:", config)
    
    # ---- wandb logging ----
    run_name = f"{config.mode}_{config.utility_kind}_alpha{config.alpha}_{config.f_divergence}"
    if config.mode == "AET":
        run_name += f"_fair_alpha{config.fair_alpha}"
    run_name += f"_seed{config.seed}"

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=run_name,
        )
    
        
    # ---- Training loop ----
    def train(agent, buffer, num_steps=10000, batch_size=256, log_interval=100):
        # stats_history = []

        for step in range(1, num_steps + 1):
            batch = buffer.sample(batch_size)
            stats = agent.train_step(batch)
            
            if step % log_interval == 0:
                print(f"[Step {step}] " + ", ".join([f"{k}: {v:.4f}" for k,v in stats.items()]))
                
                eval_results = evaluate_policy(
                    env,
                    agent,
                    num_episodes=config.num_eval_episodes,
                    config=config,
                    max_steps=config.horizon,
                    normalization_method=config.normalization_method,
                    norm_stats=norm_stats,
                    # utility=utility,
                )
                
                if config.use_wandb:
                    wandb.log({"train/" + k: v for k, v in stats.items()}, step=step)
                    wandb.log({
                        "eval/linear_scalarized_return": eval_results["linear_scalarized_return"],
                        "eval/expected_scalarized_return_piecewise_log": eval_results["expected_scalarized_return_piecewise_log"],
                        "eval/scalarized_expected_return_piecewise_log": eval_results["scalarized_expected_return_piecewise_log"],
                        "eval/primary_objective": eval_results["primary_objective"],
                    }, step=step)
    
                    # log the individual dimensions of the expected return vector
                    expected_return_vector = eval_results["expected_return_vector"]
                    return_vector_log_data = {
                        f"eval/expected_return_vector_{i}": val 
                        for i, val in enumerate(expected_return_vector)
                    }
                    wandb.log(return_vector_log_data, step=step)
                
                # print(" Train stats:", stats)
                # print(" Eval stats:", eval_results)

            # stats_history.append(stats)

        # Save final model and stats
        save_dir = os.path.join(config.save_path, run_name)
        os.makedirs(save_dir, exist_ok=True)
        agent.save(os.path.join(save_dir, "model.pth"))

    train(
        agent,
        buffer,
        num_steps= config.num_steps,
        batch_size= config.batch_size,
        log_interval=config.log_interval
    )
    if config.use_wandb:
        wandb.finish()

    
if __name__ == "__main__":
    main()
