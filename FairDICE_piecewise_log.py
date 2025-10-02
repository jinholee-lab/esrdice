# -*- coding: utf-8 -*-
# CP solver with util_type: "log" (optimize log) or "piecewise_lower" (optimize min(log,quad))
# and always post-evaluate with the true piecewise utility you proposed.

import numpy as np
import cvxpy as cp
from Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

def build_time_invariant_model(S=3, A=2, H=4, I=2, seed=7, eps=1e-8):
    rng = np.random.default_rng(seed)
    p0_raw = rng.random(S); p0 = p0_raw / p0_raw.sum()
    # 1. Initial distribution은 state에 대해서만 dependent

    # T_base[s, p, a] = T(s|p,a) (time-invariant)
    T_base = rng.random((S, S, A)) + 0.1
    T_base /= T_base.sum(axis=0, keepdims=True)
    # 2. Transition probability는 next state, state, action 순서랍니다. + 0.1은 eps랍니다. 실제로 적용시에는 좀 더 작게 0.001 정도

    # behavior policy pi_b(a|s)
    pi_raw = rng.random((S, A)) + 0.2
    pi_b = pi_raw / pi_raw.sum(axis=1, keepdims=True)
    # 3. Behavior policy는 데이터 기준으로 카운팅을한 behavior policy 입니다 + 0.2는 eps입니다. 실제로 적용시에는 좀 더 작게

    # forward flow => mu
    mu = np.zeros((S, H)); mu[:, 0] = p0
    for t in range(1, H):
        d_prev = mu[:, t-1][:, None] * pi_b   # (S,A)
        mu[:, t] = np.einsum('spa,pa->s', T_base, d_prev)

    # dataset occupancy d_D(s,a,t) = mu_t(s)*pi_b(a|s)
    d_D = np.zeros((S, A, H))
    for t in range(H):
        d_D[:, :, t] = mu[:, t][:, None] * pi_b
    d_D = np.clip(d_D, eps, None)

    # ===== time-invariant rewards: r_i(s,a) =====
    r_list = [
        0.5 + rng.random((S, A)),
        0.2 + 1.5 * rng.random((S, A)),
    ][:I]
    # 4. Reward vector는 state action에 대해서만 dependent하게 만들었습니다. reward가 발생하는 상황이 아니면 0으로 세팅하면 될 듯 합니다.

    return p0, T_base, d_D, r_list

def build_from_offline_data(dataset, S, A, H, I, eps=1e-8):
    states = dataset["states"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_states = dataset["next_states"]
    timesteps = dataset["timesteps"]
    initial_states = dataset["initial_states"]

    # 1. Initial distribution p0
    p0 = np.bincount(initial_states, minlength=S).astype(np.float64)
    p0 /= p0.sum()

    # 2. Transition probability T_base[s', s, a]
    T_base = np.zeros((S, S, A), dtype=np.float64)
    for s, a, s_next in zip(states, actions, next_states):
        T_base[s_next, s, a] += 1
    T_base += eps
    T_base /= T_base.sum(axis=0, keepdims=True)

    # 3. Dataset occupancy d_D[s, a, t]
    d_D = np.zeros((S, A, H), dtype=np.float64)
    for s, a, t in zip(states, actions, timesteps):
        d_D[s, a, t] += 1
    # normalize each timestep distribution
    for t in range(H):
        total_t = d_D[:, :, t].sum()
        if total_t > 0:
            d_D[:, :, t] /= total_t
    d_D = np.clip(d_D, eps, None)

    # 4. Reward vectors r_i(s,a)
    r_list = []
    for i in range(I):
        r_i = np.zeros((S, A), dtype=np.float64)
        counts = np.zeros((S, A), dtype=np.float64)
        for s, a, r in zip(states, actions, rewards[:, i]):
            r_i[s, a] += r
            counts[s, a] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            r_i = np.divide(r_i, counts, out=np.zeros_like(r_i), where=counts > 0)
        r_list.append(r_i)

    return p0, T_base, d_D, r_list


def piecewise_log_numpy(x, eps=1e-8):
    """Your custom piecewise utility for post-evaluation."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask = (x >= 1.0)
    out[mask] = np.log(x[mask] + eps)
    out[~mask] = -0.5 * (x[~mask] - 2.0)**2 + 0.5
    return out

def solve_cp(p0, T_base, d_D, r_list, alpha=1.0, reg_type="kl",
             util_type="log", eps=1e-8,
             solver_preference=("MOSEK","ECOS","SCS")):
    """
    reg_type: "kl" | "chi2" | "tv"
    util_type:
      - "log": optimize sum_i log(z_i)
      - "piecewise_lower": optimize sum_i y_i where y_i <= log(z_i), y_i <= quad(z_i)
    Always post-evaluates both log and true piecewise on the solution.

    NOTE: r_list contains (S,A) time-invariant rewards in this version.
    """
    S, A, H = d_D.shape
    SA = S * A

    # flatten d_D for elementwise atoms
    dD_flat = d_D.reshape(SA, H, order="C")

    # variable D: (SA x H), nonneg
    D = cp.Variable((SA, H), nonneg=True)

    def D_mat(t):
        return cp.reshape(D[:, t], (S, A), order="C")

    # constraints: init & flow
    cons = []
    cons.append(cp.sum(D_mat(0), axis=1) == p0)
    for t in range(1, H):
        lhs = cp.sum(D_mat(t), axis=1)  # (S,)
        rhs_elems = []
        d_prev_mat = D_mat(t-1)         # (S,A)
        for s in range(S):
            rhs_elems.append(cp.sum(cp.multiply(T_base[s, :, :], d_prev_mat)))
        rhs = cp.hstack(rhs_elems)
        cons.append(lhs == rhs)

    # ===== z_i = (1/H) * sum_{t} <r_i(s,a), d(s,a,t)> , with r_i(s,a) time-invariant =====
    z_list = []
    for r_i in r_list:
        r_flat = r_i.reshape(SA, order="C")[:, None]  # (SA,1)
        # inner product with each column D[:,t], then average over t
        z_i = cp.sum(cp.multiply(r_flat, D)) / H
        z_list.append(z_i)

    # utility term
    if util_type == "log":
        util_expr = cp.sum(cp.log(cp.hstack([zi for zi in z_list]) + eps))
    elif util_type == "piecewise_lower":
        # quadratic branch: q(x) = -0.5*(x-2)^2 + 0.5  (concave)
        y_vars = []
        for zi in z_list:
            yi = cp.Variable()  # scalar
            cons += [ yi <= cp.log(zi + eps) ]
            cons += [ yi <= -0.5 * cp.square(zi - 2.0) + 0.5 ]
            y_vars.append(yi)
        util_expr = cp.sum(cp.hstack(y_vars))
    else:
        raise ValueError("Unknown util_type")

    # regularizer
    if reg_type == "kl":
        reg = cp.sum(cp.kl_div(D, dD_flat))
    elif reg_type == "chi2":
        reg = 0.5 * cp.sum(cp.power(D - dD_flat, 2) / dD_flat)
    elif reg_type == "tv":
        reg = cp.sum(cp.abs(D - dD_flat))
    else:
        raise ValueError("Unknown reg_type")

    obj = cp.Maximize(util_expr - alpha * reg)
    prob = cp.Problem(obj, cons)

    last_err = None
    for sname in solver_preference:
        try:
            prob.solve(solver=getattr(cp, sname), verbose=True)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception as e:
            last_err = e
            continue
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed. status={prob.status}, last_err={last_err}")

    D_star = D.value
    d_star = D_star.reshape(S, A, H, order="C")

    # Diagnostics
    init_res = np.sum(d_star[:, :, 0], axis=1) - p0
    flow_res = []
    for t in range(1, H):
        lhs = np.sum(d_star[:, :, t], axis=1)
        rhs = np.einsum('spa,pa->s', T_base, d_star[:, :, t-1])
        flow_res.append(lhs - rhs)
    flow_res = np.stack(flow_res, axis=1) if H > 1 else np.zeros((S,0))

    # z-values and utilities (numeric)
    z_vals = []
    for r_i in r_list:
        r_flat = r_i.reshape(SA, order="C")[:, None]  # (SA,1)
        z = (np.multiply(r_flat, D_star)).sum() / H   # average over t
        z_vals.append(z)
    z_vals = np.array(z_vals, dtype=np.float64)

    # post-evaluate objectives
    log_util_post = np.sum(np.log(z_vals + eps))
    piecewise_util_post = np.sum(piecewise_log_numpy(z_vals, eps=eps))

    # reg value (numeric)
    if reg_type == "kl":
        reg_val = np.sum(D_star * (np.log(np.maximum(D_star, 1e-300)) - np.log(dD_flat)) - D_star + dD_flat)
    elif reg_type == "chi2":
        reg_val = 0.5 * np.sum((D_star - dD_flat)**2 / dD_flat)
    else:
        reg_val = np.sum(np.abs(D_star - dD_flat))

    # recovered policy pi*(a|s,t)
    s_marg = d_star.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pi_star = np.divide(d_star, s_marg, out=np.zeros_like(d_star), where=(s_marg > 0))

    return {
        "status": prob.status,
        "util_type": util_type,
        "z_values": z_vals.tolist(),
        "log_util_post": float(log_util_post),
        "piecewise_util_post": float(piecewise_util_post),
        "regularizer_value": float(reg_val),
        "max_init_residual": float(np.abs(init_res).max()),
        "max_flow_residual": float(np.abs(flow_res).max()) if flow_res.size else 0.0,
        "pi_star": pi_star,
        "d_star": d_star,
    }

from utils import Utility

def evaluate_policy_cp(
    env,
    pi_star,
    num_episodes=10,
    max_steps=40,
    normalization_method="linear",
    norm_stats=None,
    utility=None,
    rollout_mode="stochastic",   # or "deterministic"
):
    """
    Evaluate a convex-program optimized policy (pi_star) in the environment.

    Args:
        env: Taxi environment
        pi_star: np.array of shape (S, A, H), from solve_cp
        num_episodes: number of episodes for evaluation
        max_steps: rollout horizon (should match H used in solver)
        normalization_method: "linear" or "none"
        norm_stats: normalization statistics (if used during training)
        utility: Utility instance for scalarized evaluation
        rollout_mode: "stochastic" (sample) or "deterministic" (argmax)

    Returns:
        dict: evaluation results (expected returns, scalarized returns, etc.)
    """
    return_vector_list = []
    scalarized_return_list = []

    S, A, H = pi_star.shape

    for ep in range(num_episodes):
        state = env.reset(taxi_loc=[5, 4])   # or sample initial condition
        done = False
        step = 0
        Racc = np.zeros(len(env.loc_coords), dtype=np.float32)
        ep_rewards = []

        while not done and step < max_steps:
            # --- choose action from pi_star ---
            if step >= H:
                # if rollout exceeds optimization horizon, use final policy
                action_probs = pi_star[state, :, -1]
            else:
                action_probs = pi_star[state, :, step]

            if rollout_mode == "stochastic":
                action = np.random.choice(A, p=action_probs)
            else:
                action = np.argmax(action_probs)

            # --- environment step ---
            next_state, original_reward_vec, done = env.step(action)
            original_reward_vec = np.array(original_reward_vec, dtype=np.float32)

            # --- normalization ---
            if normalization_method == "linear":
                if norm_stats is None:
                    raise ValueError("norm_stats must be provided for normalization.")
                normalized_reward_vec = original_reward_vec / norm_stats["rewards"]["max"]
            else:
                normalized_reward_vec = original_reward_vec

            # --- update cumulative reward ---
            Racc += normalized_reward_vec
            ep_rewards.append(normalized_reward_vec)

            state = next_state
            step += 1

        # --- episode summary ---
        one_episode_return_vector = np.sum(ep_rewards, axis=0)
        return_vector_list.append(one_episode_return_vector)

        if utility is not None:
            scalarized_val = utility(one_episode_return_vector)
        else:
            scalarized_val = one_episode_return_vector.sum()

        scalarized_return_list.append(scalarized_val)

    # Convert to arrays
    return_vector_array = np.array(return_vector_list)
    scalarized_return_array = np.array(scalarized_return_list)

    # Metrics
    expected_return_vector = return_vector_array.mean(axis=0)
    linear_scalarized_return = expected_return_vector.sum()
    expected_scalarized_return = scalarized_return_array.mean()
    scalarized_expected_return = utility(expected_return_vector) if utility is not None else expected_return_vector.sum()

    print(f"Expected Return Vector: {expected_return_vector}")
    print(f"Linear Scalarized Return: {linear_scalarized_return}")
    print(f"Expected Scalarized Return: {expected_scalarized_return}")
    print(f"Scalarized Expected Return: {scalarized_expected_return}")

    return {
        "expected_return_vector": expected_return_vector,
        "linear_scalarized_return": linear_scalarized_return,
        "expected_scalarized_return": expected_scalarized_return,
        "scalarized_expected_return": scalarized_expected_return,
    }

if __name__ == "__main__":
    # Build test instance
    # S, A, H, I = 30, 4, 10, 2
    # p0, T_base, d_D, r_list = build_time_invariant_model(S, A, H, I, seed=7, eps=1e-8)
    
    size = 10
    loc_coords = [[0,2],[9,7]]
    dest_coords = [[0,0],[9,9]]
    fuel = 40

    env = Fair_Taxi_MDP_Penalty_V2(
        size=size,
        loc_coords=loc_coords,
        dest_coords=dest_coords,
        fuel=fuel,
        output_path=None
    )
    
    dataset = np.load("./data/bus.npy", allow_pickle=True).item()
    S = env.observation_space.n       # 600
    A = env.action_space.n            # 6
    H = 40
    I = 2

    p0, T_base, d_D, r_list = build_from_offline_data(dataset, S, A, H, I)


    # Choose utility (see notes above)
    util_type = "piecewise_lower"   # "log" or "piecewise_lower"
    reg_type  = "chi2"              # "kl" | "chi2" | "tv"
    alpha     = 0.1

    rep = solve_cp(p0, T_base, d_D, r_list,
                   alpha=alpha, reg_type=reg_type, util_type=util_type)

    np.set_printoptions(precision=4, suppress=True)
    print("\n=== Solve Summary ===")
    for k in ["status","util_type","z_values","log_util_post","piecewise_util_post",
              "regularizer_value","max_init_residual","max_flow_residual"]:
        print(f"{k}: {rep[k]}")
    print("\npi*(a|s,t) at t=0:\n", rep["pi_star"][:, :, 0])
    print("pi*(a|s,t) at t=H-1:\n", rep["pi_star"][:, :, -1])
    
    pi_star = rep["pi_star"]  # shape: (S, A, H)
    
    np.save("./pi_star_piecewise_log.npy", pi_star)

    results = evaluate_policy_cp(
        env=env,
        pi_star=pi_star,
        num_episodes=20,
        max_steps=40,
        normalization_method="linear",
        norm_stats=None,   # same stats used during training
        rollout_mode="stochastic"
    )
    
    