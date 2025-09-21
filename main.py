

import argparse
import numpy as np
import torch

from enum import Enum
from pathlib import Path
from utils import *
from buffer import ReplayBuffer
from finitedice import FiniteOptiDICE
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2
from evaluate import evaluate_policy
from divergence import FDivergence

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
        print("[Warning] No normalization applied.")
        pass

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


def preprocess_scalarization(dataset, utility):
    """
    Scalarization된 ESR 보상 계산 후 dataset에 추가

    Args:
      dataset: dict
        - 'Raccs' : [N, D]
        - 'rewards' : [N, D]
      utility: Utility 객체 (u(R))

    Returns:
      new_dataset: dataset에 'rewards_esr' 추가
    """
    Raccs   = dataset["Raccs"]      # [N, D]
    rewards = dataset["rewards"]    # [N, D]

    # ESR step-wise reward
    esr = utility(Raccs + rewards) - utility(Raccs)   # [N]
    esr = esr.astype(np.float32)[:, None]             # [N,1]

    new_dataset = dataset.copy()
    new_dataset["rewards"] = esr
    return new_dataset

def main():
    parser = argparse.ArgumentParser()

    # ---- Experiment setup ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=100)

    # ---- Env setup ----
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--fuel", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="./outputs/")
    parser.add_argument("--reward_dim", type=int, default=2)
    parser.add_argument("--utility_kind", type=str, default="piecewise_log")

    # ---- OptiDICE setup ----
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--time_embed_dim", type=int, default=8)
    parser.add_argument("--layer_norm", type=bool, default=True)

    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--nu_lr", type=float, default=3e-4)
    parser.add_argument("--f_divergence", type=str, default="Chi",
                        choices=[d.value for d in FDivergence])
    parser.add_argument("--nu_grad_penalty_coeff", type=float, default=0.001)

    # ---- Dataset / Replay buffer ----
    parser.add_argument("--dataset_path", type=str, default="./data/dataset_v3.npy")
    parser.add_argument("--reward_index", type=int, default=0)
    parser.add_argument("--one_hot_pass_idx", type=bool, default=True)
    parser.add_argument("--concat_acc_reward", type=bool, default=True)
    parser.add_argument("--normalization", type=bool, default=False)
    parser.add_argument("--normalization_method", type=str, default="linear", choices=["zscore", "minmax", "linear"])

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
    
    # ---- Env init ----
    size, loc_coords, dest_coords = get_setting(args.size, args.reward_dim)

    env = Fair_Taxi_MDP_Penalty_V2(
        size=size,
        loc_coords=loc_coords,
        dest_coords=dest_coords,
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
    utility = Utility(kind=config.utility_kind, weights=None, shift=0.0)
    dataset = preprocess_scalarization(dataset, utility)
    
    # ---- Preprocess states ----
    dataset = preprocess_states(
        dataset,
        env,
        one_hot_xy=False,
        one_hot_pass_idx=config.one_hot_pass_idx,
        concat_raccs=config.concat_acc_reward,
    )
    
    config.state_dim = dataset["states"].shape[1]
    config.action_dim = env.action_space.n
    config.reward_dim = dataset["rewards"].shape[1]  # scalarized reward
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}, Reward dim: {config.reward_dim}")
    
    # ---- Replay Buffer ----
    buffer = ReplayBuffer(device=config.device)
    buffer.load_dataset(dataset)

    # ---- Initialize OptiDICE ----
    config.f_divergence = FDivergence(config.f_divergence)
    agent = FiniteOptiDICE(config, device=config.device)

    print("✅ Environment and agent initialized.")
    print("Config:", config)
    # ---- Training loop ----
    def train(agent, buffer, num_steps=10000, batch_size=256, log_interval=100):
        stats_history = []

        for step in range(1, num_steps + 1):
            batch = buffer.sample(batch_size)
            stats = agent.train_step(batch)
            
            if step % log_interval == 0:
                print(f"[Step {step}] " + ", ".join([f"{k}: {v:.4f}" for k,v in stats.items()]))
                eval_results = evaluate_policy(
                    env,
                    agent,
                    num_episodes=10,
                    config=config,
                    max_steps=config.horizon,
                    normalization_method=config.normalization_method,
                    norm_stats=norm_stats,
                    utility=utility                # ESR 학습에 썼던 Utility
                )
            stats_history.append(stats)

        return stats_history

    stats_history = train(
        agent,
        buffer,
        num_steps= config.num_steps,
        batch_size= config.batch_size,
        log_interval=config.log_interval
    )

    
if __name__ == "__main__":
    main()
