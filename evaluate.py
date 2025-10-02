import numpy as np
import torch

from utils import Utility

def evaluate_policy(
    env,
    agent,
    config,
    num_episodes=10,
    max_steps=100,
    normalization_method="linear",
    norm_stats=None,
    utility=None,
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
    if config.utility_kind == "linear":
        # set utility as piecewise_log with uniform weights
        weights = np.ones(config.reward_dim, dtype=np.float32)
        utility = Utility(kind="piecewise_log", weights=weights)
    
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

            # --- Raccs 붙이기 ---
            if getattr(config, "concat_acc_reward", False):
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
            else:
                # no normalization
                normalized_reward_vec = original_reward_vec
                        
            # --- update cumulative reward ---
            Racc += normalized_reward_vec

            state = next_state
            step += 1
            
            ep_rewards.append(normalized_reward_vec)


        # --- episode summary --- 
        one_episode_return_vector = np.sum(ep_rewards, axis=0)
        return_vector_list.append(one_episode_return_vector)

        if utility is not None:
            one_episode_scalarized_return = utility(one_episode_return_vector)
        else:
            one_episode_scalarized_return = one_episode_return_vector.sum()  # fallback: 단순합
    
        scalarized_return_list.append(one_episode_scalarized_return)

    scalarized_return_array = np.array(scalarized_return_list) # episode level
    return_vector_array = np.array(return_vector_list) # episode level

    # Expected vector return
    expected_return_vector = return_vector_array.mean(axis=0)
    print(f"Expected Return Vector: {expected_return_vector}")

    # Linear scalarization of expected return vector
    linear_scalarized_return = expected_return_vector.sum()
    print(f"Linear Scalarized Return: {linear_scalarized_return}")

    # Expected scalarized return
    expected_scalarized_return = scalarized_return_array.mean()
    print(f"Expected Scalarized Return: {expected_scalarized_return}")
    
    # Scalarized expected return
    scalarized_expected_return = utility(expected_return_vector) if utility is not None else expected_return_vector.sum()
    print(f"Scalarized Expected Return: {scalarized_expected_return}")


    results = {
        "expected_return_vector": expected_return_vector,
        "linear_scalarized_return": linear_scalarized_return,
        f"expected_scalarized_return_{config.utility_kind}": expected_scalarized_return,
        f"scalarized_expected_return_{config.utility_kind}": scalarized_expected_return,
    }
    return results
