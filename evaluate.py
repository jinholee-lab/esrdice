import numpy as np
import torch

def evaluate_policy(
    env,
    agent,
    config,
    num_episodes=50,
    max_steps=100,
    render=False,
    normalize=False,
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
    returns_all_scalar = []
    returns_all_vector = []

    for ep in range(num_episodes):
        state = env.reset(taxi_loc = [9,9])
        done = False
        step = 0
        Racc = np.zeros(2, dtype=np.float32)  # 누적 보상 (정규화 전/후)
        # print(Racc.shape)
        ep_rewards = []

        while not done and step < max_steps:
            # state 전처리 (학습 config와 동일)
            s_decoded = list(env.decode(state))
            if getattr(config, "one_hot_pass_idx", False):
                one_hot = np.zeros(len(env.dest_coords) + 1, dtype=np.float32)
                one_hot[s_decoded[3]] = 1.0
                s_feat = np.array([s_decoded[0], s_decoded[1], s_decoded[2]] + one_hot.tolist(), dtype=np.float32)
            else:
                s_feat = np.array(s_decoded, dtype=np.float32)

            if getattr(config, "concat_acc_reward", False):
                # print(f"passed this{1}")
                s_feat = np.concatenate([s_feat, Racc], axis=0)

            s_tensor = torch.tensor(s_feat, dtype=torch.float32, device=agent.device).unsqueeze(0)
            # print(s_tensor.shape)
            with torch.no_grad():
                dist = agent.policy(s_tensor)
                action = dist.probs.argmax(dim=-1).item()

            next_state, reward_vec, done = env.step(action)
            reward_vec = np.array(reward_vec, dtype=np.float32)

            # --- normalization (if used during training) ---
            if normalize and norm_stats is not None and "rewards" in norm_stats:
                stat = norm_stats["rewards"]
                min_val, max_val = stat["min"], stat["max"]
                if max_val > min_val:
                    normalzied_reward_vec = (reward_vec - min_val) / (max_val - min_val)

            # --- update cumulative reward ---
            Racc += normalzied_reward_vec
            ep_rewards.append(reward_vec)

            if render:
                env.render()

            state = next_state
            step += 1

        # --- episode summary ---
        total_vector = np.sum(ep_rewards, axis=0)
        returns_all_vector.append(total_vector)

        if utility is not None:
            total_scalar = utility(total_vector)
        else:
            total_scalar = total_vector.sum()  # fallback: 단순합
        returns_all_scalar.append(total_scalar)

        print(f"[Eval Ep {ep+1}] Steps={step}, Vector Return={total_vector}, Scalar={total_scalar}")

    returns_all_scalar = np.array(returns_all_scalar)
    returns_all_vector = np.array(returns_all_vector)

    results = {
        "avg_scalar_return": returns_all_scalar.mean(),
        "std_scalar_return": returns_all_scalar.std(),
        "avg_vector_return": returns_all_vector.mean(axis=0),
        "std_vector_return": returns_all_vector.std(axis=0),
        "all_scalar_returns": returns_all_scalar,
        "all_vector_returns": returns_all_vector,
    }
    return results
