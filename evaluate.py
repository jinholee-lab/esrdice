import numpy as np
import torch

from utility import Utility
from utils import Utility_np as Vector_Utility

def _get_policy(agent, policy_name: str):
    """
    Resolve policy object from agent by name.
    """
    if not hasattr(agent, policy_name):
        raise AttributeError(
            f"Agent has no policy named '{policy_name}'. "
            f"Available: {[k for k in dir(agent) if k.startswith('policy')]}"
        )
    return getattr(agent, policy_name)


def evaluate_policy(
    env,
    agent,
    config,
    num_episodes=10,
    max_steps=100,
    normalization_method="linear",
    norm_stats=None,
):
    """
    Evaluate trained policy with normalization and utility.

    Args:
        env: Taxi environment
        agent: trained FiniteOptiDICE agent
        num_episodes: number of evaluation episodes
        max_steps: rollout horizon
        render: visualize with pygame
        normalize: whether to normalize rewards (same as training)
        norm_stats: dict of normalization statistics (from normalize_dataset)
        utility: Utility instance for scalarized evaluation

    Returns:
        dict with results
    """
    agent.policy.eval()
    return_vector_list = []
    scalarized_return_list = []
    return_vector_list_of_primary_objective = []

    eval_utility = Utility(kind="piecewise_log")
    if config.utility_kind == "alpha_fairness":
        vector_transform_utility = Utility(kind="alpha_fairness", alpha=1 - config.fair_alpha)
        scalarization_utility = Utility(kind="alpha_fairness", alpha=config.fair_alpha)

    for ep in range(num_episodes):
        # state = env.reset(taxi_loc = [9,9])
        # state = env.reset(taxi_loc = [4, 5])
        state = env.reset(taxi_loc = [5,4])
        done = False
        step = 0
        Racc = np.zeros(len(config.loc_coords), dtype=np.float32)  # accumulated reward vector
        ep_rewards = []

        while not done and step < max_steps:
            # state preprocessing
            s_decoded = list(env.decode(state))
            taxi_x, taxi_y, pass_loc, pass_idx = s_decoded
            feats = []

            # --- Taxi 위치 ---
            if getattr(config, "one_hot_xy", False):
                taxi_x_oh = np.eye(env.size)[taxi_x]
                taxi_y_oh = np.eye(env.size)[taxi_y]
                feats.append(taxi_x_oh)
                feats.append(taxi_y_oh)
            else:
                feats.append([taxi_x])
                feats.append([taxi_y])

            # --- Passenger 상태 ---
            feats.append([pass_loc])

            if getattr(config, "one_hot_pass_idx", False):
                pass_idx_oh = np.eye(len(env.dest_coords) + 1)[pass_idx]
                feats.append(pass_idx_oh)
            else:
                feats.append([pass_idx])

            feats = np.concatenate(feats, axis=0)

            # --- Raccs ---
            # if getattr(config, "concat_acc_reward", False):
            feats = np.concatenate([feats, Racc], axis=0)

            s_tensor = torch.tensor(feats, dtype=torch.float32, device=agent.device).unsqueeze(0)
            t_tensor = torch.tensor([[step]], dtype=torch.long, device=agent.device)
            with torch.no_grad():
                dist = agent.policy(s_tensor, t_tensor)  # [1, A]

                if getattr(config, "policy_rollout", "stochastic") == "stochastic":
                    # stochastic
                    action = dist.sample().item()
                else:
                    # deterministic
                    action = dist.probs.argmax(dim=-1).item()

            next_state, original_reward_vec, done = env.step(action)
            original_reward_vec = np.array(original_reward_vec, dtype=np.float32)

            # --- normalization (if used during training) ---
            if normalization_method == "linear":
                if norm_stats is None:
                    raise ValueError("norm_stats must be provided for normalization.")
                normalized_reward_vec = original_reward_vec / norm_stats["rewards"]["max"]
            elif normalization_method == "minmax":
                normalized_reward_vec = original_reward_vec / norm_stats["rewards"]["max"] #TODO: check
            else:
                # no normalization
                normalized_reward_vec = original_reward_vec

            # --- update cumulative reward ---
            Racc += normalized_reward_vec

            state = next_state
            step += 1

            ep_rewards.append(normalized_reward_vec)


        # --- episode summary ---
        ep_return_vector = np.sum(ep_rewards, axis=0)

        ep_return_vector_tensor = torch.as_tensor(ep_return_vector, dtype=torch.float32)
        ep_scalarized_return_tensor = eval_utility(ep_return_vector_tensor, keep_dims=False)
        ep_primary_objective_scalarized_return_tensor = vector_transform_utility(ep_return_vector_tensor, keep_dims=True)

        return_vector_list.append(ep_return_vector_tensor)
        scalarized_return_list.append(ep_scalarized_return_tensor)
        return_vector_list_of_primary_objective.append(ep_primary_objective_scalarized_return_tensor)
        # --- end of episode ---

    scalarized_return_tensor = torch.stack(scalarized_return_list)  # episode level
    return_vector_tensor = torch.stack(return_vector_list)  # episode level
    return_vector_tensor_of_primary_objective = torch.stack(return_vector_list_of_primary_objective)  # episode level

    # Expected vector return
    expected_return_vector = return_vector_tensor.mean(dim=0)
    print(f"Expected Return Vector: {expected_return_vector}")

    # Linear scalarization of expected return vector
    linear_scalarized_return = expected_return_vector.sum().item()
    print(f"Linear Scalarized Return: {linear_scalarized_return}")

    # Expected scalarized return
    expected_scalarized_return = scalarized_return_tensor.mean().item()
    print(f"Expected Scalarized Return {eval_utility.kind}: {expected_scalarized_return}")

    # Scalarized expected return
    scalarized_expected_return = eval_utility(expected_return_vector).item()
    print(f"Scalarized Expected Return {eval_utility.kind}: {scalarized_expected_return}")

    # Primary objective
    expected_return_vector_of_primary_objective = return_vector_tensor_of_primary_objective.mean(dim=0)
    print(f"Expected Return Vector of Primary Objective: {expected_return_vector_of_primary_objective}")
    scalarized_expected_return_of_primary_objective = scalarization_utility(expected_return_vector_of_primary_objective, keep_dims=False)
    print(f"Scalarized Expected Return of Primary Objective: {scalarized_expected_return_of_primary_objective}")

    results = {
        "expected_return_vector": expected_return_vector,
        "linear_scalarized_return": linear_scalarized_return,
        f"expected_scalarized_return_{eval_utility.kind}": expected_scalarized_return,
        f"scalarized_expected_return_{eval_utility.kind}": scalarized_expected_return,
        # "expected_return_vector_of_primary_objective": expected_return_vector_of_primary_objective,
        "primary_objective": scalarized_expected_return_of_primary_objective,
    }
    return results


def evaluate_policy_vector(
    env,
    agent,
    policy_name,           # <<< NEW (e.g. "policy_1" or "policy_2")
    config,
    num_episodes=10,
    max_steps=100,
    normalization_method="linear",
    norm_stats=None,
    utility=None,
):
    """
    Evaluate a specified policy of FiniteAET_double.

    Args:
        env: environment
        agent: FiniteAET_double
        policy_name: str, one of {"policy_1", "policy_2"}
        config: argparse config
        num_episodes: number of evaluation episodes
        max_steps: rollout horizon
        normalization_method: reward normalization method
        norm_stats: normalization statistics
        utility: optional Utility (vector-form)

    Returns:
        dict with evaluation metrics
    """

    policy = _get_policy(agent, policy_name)
    policy.eval()

    return_vector_list = []
    scalarized_return_list = []
    return_vector_list_of_primary_objective = []

    eval_utility = Vector_Utility(kind="piecewise_log")

    if config.utility_kind == "alpha_fairness":
        vector_transform_utility = Vector_Utility(
            kind="alpha_fairness",
            alpha=1 - config.fair_alpha
        )
        scalarization_utility = Vector_Utility(
            kind="alpha_fairness",
            alpha=config.fair_alpha
        )

    for ep in range(num_episodes):
        state = env.reset(taxi_loc=[5, 4])
        done = False
        step = 0

        Racc = np.zeros(len(config.loc_coords), dtype=np.float32)
        ep_rewards = []

        while not done and step < max_steps:
            # --- decode state ---
            taxi_x, taxi_y, pass_loc, pass_idx = env.decode(state)

            feats = []

            # taxi position
            if getattr(config, "one_hot_xy", False):
                feats.append(np.eye(env.size)[taxi_x])
                feats.append(np.eye(env.size)[taxi_y])
            else:
                feats.append([taxi_x])
                feats.append([taxi_y])

            # passenger
            feats.append([pass_loc])
            if getattr(config, "one_hot_pass_idx", False):
                feats.append(np.eye(len(env.dest_coords) + 1)[pass_idx])
            else:
                feats.append([pass_idx])

            feats = np.concatenate(feats, axis=0)

            # accumulated rewards
            feats = np.concatenate([feats, Racc], axis=0)

            s_tensor = torch.tensor(
                feats, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)

            t_tensor = torch.tensor(
                [[step]], dtype=torch.long, device=agent.device
            )

            with torch.no_grad():
                dist = policy.get_dist(s_tensor, t_tensor)

                if getattr(config, "policy_rollout", "stochastic") == "stochastic":
                    action = dist.sample().item()
                else:
                    action = dist.logits.argmax(dim=-1).item()

            next_state, reward_vec, done = env.step(action)
            reward_vec = np.asarray(reward_vec, dtype=np.float32)

            # normalization
            if normalization_method == "linear":
                reward_vec = reward_vec / norm_stats["rewards"]["max"]
            elif normalization_method == "minmax":
                reward_vec = reward_vec / norm_stats["rewards"]["max"]
            # else: no normalization

            Racc += reward_vec
            ep_rewards.append(reward_vec)

            state = next_state
            step += 1

        # --- episode summary ---
        ep_return_vector = np.sum(ep_rewards, axis=0)
        return_vector_list.append(ep_return_vector)

        ep_scalarized_return = eval_utility(ep_return_vector)
        scalarized_return_list.append(ep_scalarized_return)

        ep_primary_vector = vector_transform_utility(
            ep_return_vector, keep_dims=True
        )
        return_vector_list_of_primary_objective.append(ep_primary_vector)

    return_vector_array = np.asarray(return_vector_list)
    scalarized_return_array = np.asarray(scalarized_return_list)
    primary_vector_array = np.asarray(return_vector_list_of_primary_objective)

    expected_return_vector = return_vector_array.mean(axis=0)
    linear_scalarized_return = expected_return_vector.sum()
    expected_scalarized_return = scalarized_return_array.mean()
    scalarized_expected_return = eval_utility(expected_return_vector)

    expected_primary_vector = primary_vector_array.mean(axis=0)
    primary_objective = scalarization_utility(
        expected_primary_vector, keep_dims=False
    )

    results = {
        "expected_return_vector": expected_return_vector,
        "linear_scalarized_return": linear_scalarized_return,
        f"expected_scalarized_return_{eval_utility.kind}": expected_scalarized_return,
        f"scalarized_expected_return_{eval_utility.kind}": scalarized_expected_return,
        "primary_objective": primary_objective,
    }

    return results

