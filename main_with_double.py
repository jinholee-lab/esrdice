# main_with_double.py (updated for FiniteAET_double with policy_1 / policy_2)
# Minimal changes:
#  - call evaluate_policy_vector(..., policy_name="policy_1"/"policy_2")
#  - log eval results separately (eval_p1/*, eval_p2/*) for clarity

import argparse
import os

import numpy as np
import torch

import wandb
from AET_double import FiniteAET_double
from buffer import ReplayBuffer
from divergence import FDivergence
from evaluate import evaluate_policy_vector
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2
from utility import Utility
from utils import get_setting


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
        raise KeyError("rewards is not in dataset")

    arr = dataset["rewards"]
    stat_dict = {}

    if method == "linear":
        max_val = arr.max()
        if max_val > 0:
            new_dataset["rewards"] = arr / max_val
        else:
            raise ValueError("max reward is non-positive, please check the dataset")
        stat_dict["rewards"] = {"max": max_val}
    elif method == "zscore":
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            new_dataset["rewards"] = (arr - mean) / std
        else:
            new_dataset["rewards"] = arr
        stat_dict["rewards"] = {"mean": mean, "std": std}
    elif method == "minmax":
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            new_dataset["rewards"] = (arr - min_val) / (max_val - min_val)
        else:
            new_dataset["rewards"] = arr
        stat_dict["rewards"] = {"max": max_val, "min": min_val}
    else:
        new_dataset["rewards"] = arr
        max_val = arr.max()
        stat_dict["rewards"] = {"max": max_val}

    return new_dataset, stat_dict


def preprocess_Raccs(dataset, horizon):
    rewards = dataset["rewards"]  # [N, D]
    N, D = rewards.shape

    Raccs = np.zeros((N, D), dtype=np.float32)
    next_Raccs = np.zeros((N, D), dtype=np.float32)

    final_Raccs_list = []

    for start in range(0, N, horizon):
        acc = np.zeros(D, dtype=np.float32)
        for t in range(horizon):
            idx = start + t
            Raccs[idx] = acc
            acc = acc + rewards[idx]
            next_Raccs[idx] = acc
        final_Raccs_list.append(acc.copy())
    final_Raccs_array = np.array(final_Raccs_list)

    new_dataset = dataset.copy()
    new_dataset["Raccs"] = Raccs
    new_dataset["next_Raccs"] = next_Raccs
    return new_dataset, final_Raccs_array


def preprocess_states(
    dataset,
    env,
    one_hot_xy=False,
    one_hot_pass_idx=True,
    concat_raccs=True,
):
    def decode_and_feat(states, raccs=None):
        decoded = np.array([env.decode(s) for s in states])  # (N,4)
        taxi_x, taxi_y, pass_loc, pass_idx = decoded.T

        feats = []

        if one_hot_xy:
            taxi_x_oh = np.eye(env.size)[taxi_x]
            taxi_y_oh = np.eye(env.size)[taxi_y]
            feats.append(taxi_x_oh)
            feats.append(taxi_y_oh)
        else:
            feats.append(taxi_x[:, None])
            feats.append(taxi_y[:, None])

        feats.append(pass_loc[:, None])

        if one_hot_pass_idx:
            pass_idx_oh = np.eye(len(env.dest_coords) + 1)[pass_idx]
            feats.append(pass_idx_oh)
        else:
            feats.append(pass_idx[:, None])

        feats = np.concatenate(feats, axis=1)

        if concat_raccs and raccs is not None:
            feats = np.concatenate([feats, raccs], axis=1)

        return feats

    new_dataset = dataset.copy()
    new_dataset["states"] = decode_and_feat(dataset["states"], dataset.get("Raccs"))
    new_dataset["next_states"] = decode_and_feat(dataset["next_states"], dataset.get("next_Raccs"))
    new_dataset["initial_states"] = decode_and_feat(
        dataset["initial_states"], np.zeros_like(dataset["Raccs"])
    )

    return new_dataset


def preprocess_scalarization(dataset, utility, keep_dims=False, horizon=100, add_u0_to="first"):
    Raccs = dataset["Raccs"]
    rewards = dataset["rewards"]
    N, D = rewards.shape

    esr = utility(Raccs + rewards, keep_dims=keep_dims) - utility(Raccs, keep_dims=keep_dims)

    if horizon is not None:
        if N % horizon != 0:
            raise ValueError(f"N={N} is not a multiple of horizon={horizon}.")

        if keep_dims:
            u0 = utility(np.zeros((1, D), dtype=np.float32), keep_dims=True)[0]
        else:
            u0 = utility(np.zeros((1, D), dtype=np.float32), keep_dims=False)[0]

        if add_u0_to == "first":
            idx = np.arange(0, N, horizon)
        elif add_u0_to == "last":
            idx = np.arange(horizon - 1, N, horizon)
        else:
            raise ValueError("add_u0_to must be 'first' or 'last'.")

        esr[idx] += u0

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
    parser.add_argument("--f_divergence", type=str, default="Chi", choices=[d.value for d in FDivergence])
    parser.add_argument("--nu_grad_penalty_coeff", type=float, default=0.001)
    parser.add_argument("--fair_alpha", type=float, default=1.0)
    parser.add_argument("--lambda_1", type=float, default=None)
    parser.add_argument("--lambda_2", type=float, default=None)

    # ---- path ----
    parser.add_argument("--save_path", type=str, default="./checkpoints/")

    # ---- Dataset / Replay buffer ----
    parser.add_argument("--dataset_path", type=str, default="./data/dataset_v3.npy")
    parser.add_argument("--one_hot_xy", type=bool, default=False)
    parser.add_argument("--one_hot_pass_idx", type=bool, default=False)
    parser.add_argument("--concat_acc_reward", type=bool, default=False)
    parser.add_argument(
        "--normalization_method",
        type=str,
        default="linear",
        choices=["zscore", "minmax", "linear", "none"],
    )
    parser.add_argument("--use_augmentation", type=bool, default=True)

    # ---- logging ----
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="AET")
    parser.add_argument("--tag", type=str, default="test")

    # ---- mode ----
    parser.add_argument("--policy_rollout", type=str, default="deterministic", choices=["stochastic", "deterministic"])

    args = parser.parse_args()

    config = args
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    keep_dims = True  # Vector-form Actor / Critic

    # ---- Env init ----
    config.size, config.loc_coords, config.dest_coords = get_setting(args.size, args.reward_dim)

    env = Fair_Taxi_MDP_Penalty_V2(
        size=config.size,
        loc_coords=config.loc_coords,
        dest_coords=config.dest_coords,
        fuel=config.fuel,
        output_path=config.output_path,
    )

    # ---- Load dataset ----
    dataset = np.load(config.dataset_path, allow_pickle=True).item()

    dataset = preprocess_timesteps(dataset)
    dataset, norm_stats = preprocess_rewards(dataset, method=config.normalization_method)
    dataset, final_Raccs_array = preprocess_Raccs(dataset, horizon=config.horizon)

    esr_utility = Utility(kind=config.utility_kind, alpha=1 - config.fair_alpha)

    dataset = preprocess_states(
        dataset,
        env,
        one_hot_xy=config.one_hot_xy,
        one_hot_pass_idx=config.one_hot_pass_idx,
        concat_raccs=config.concat_acc_reward,
    )

    config.state_dim = dataset["states"].shape[1] + (config.reward_dim if not config.concat_acc_reward else 0)
    config.action_dim = env.action_space.n
    config.reward_dim = dataset["rewards"].shape[1]
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}, Reward dim: {config.reward_dim}")

    # ---- Replay Buffer ----
    buffer = ReplayBuffer(
        device=config.device,
        utility=esr_utility,
        horizon=config.horizon,
        keep_dims=keep_dims,
        reward_dim=config.reward_dim,
        use_augmentation=config.use_augmentation,
    )
    buffer.load_dataset(dataset)

    # ---- Initialize DICE ----
    config.f_divergence = FDivergence(config.f_divergence)

    print("✅ Environment and agent initialized.")
    print("Config:", config)

    def trainer(agent, buffer, num_steps=10000, batch_size=256, log_interval=100):
        for step in range(1, num_steps + 1):
            batch = buffer.sample(batch_size)
            stats = agent.train_step(batch)

            if step % log_interval == 0:
                print(f"[Step {step}] " + ", ".join([f"{k}: {v:.4f}" for k, v in stats.items()]))

                # Evaluate BOTH policies
                eval_p1 = evaluate_policy_vector(
                    env,
                    agent,
                    policy_name="policy_1",
                    num_episodes=config.num_eval_episodes,
                    config=config,
                    max_steps=config.horizon,
                    normalization_method=config.normalization_method,
                    norm_stats=norm_stats,
                )

                eval_p2 = evaluate_policy_vector(
                    env,
                    agent,
                    policy_name="policy_2",
                    num_episodes=config.num_eval_episodes,
                    config=config,
                    max_steps=config.horizon,
                    normalization_method=config.normalization_method,
                    norm_stats=norm_stats,
                )

                if config.use_wandb:
                    wandb.log({"train/" + k: v for k, v in stats.items()}, step=step)

                    wandb.log({f"eval_p1/{k}": v for k, v in eval_p1.items()}, step=step)
                    wandb.log({f"eval_p2/{k}": v for k, v in eval_p2.items()}, step=step)

                    # log individual dimensions of expected return vector (both policies)
                    for i, val in enumerate(eval_p1["expected_return_vector"]):
                        wandb.log({f"eval_p1/expected_return_vector_{i}": float(val)}, step=step)

                    for i, val in enumerate(eval_p2["expected_return_vector"]):
                        wandb.log({f"eval_p2/expected_return_vector_{i}": float(val)}, step=step)

    # ---- wandb logging ----
    model_name = f"Postfilter_fair_alpha{config.fair_alpha}"
    model_name += f"_filter{config.lambda_1}"
    model_name += f"_budget{config.lambda_2}"
    model_name += f"_seed{config.seed}"
    tag_name = [config.tag] if isinstance(config.tag, str) else config.tag

    save_model_dir = os.path.join(config.save_path, model_name)
    model_path = os.path.join(save_model_dir, "model.pth")

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=model_name,
            tags=tag_name,
        )

    agent = FiniteAET_double(config=config, device=config.device)
    trainer(
        agent,
        buffer,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        log_interval=config.log_interval,
    )

    os.makedirs(save_model_dir, exist_ok=True)
    agent.save(model_path)
    print(f"💾 모델 저장 완료: {model_path}")

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
