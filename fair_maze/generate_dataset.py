import os
import imageio  # optional, only if you still want GIFs
import numpy as np
import torch
from rlkit.torch.pytorch_util import set_gpu_mode

from fair_maze_env import FairAntMaze, WW, EM, SP, O1, D1, O2, D2
from wrappers import NormalizedBoxEnv

import argparse

# ----------------- Policy loading -----------------

POLICY_FILE = os.path.join(
    os.path.dirname(__file__), "..", "fair_maze", "policies", "ant_hierarch_pol.pkl"
)

def load_policy(policy_file):
    data = torch.load(policy_file, map_location="cuda")
    print("policy file keys:", data.keys())
    policy = data["exploration/policy"]
    env = data["evaluation/env"]
    print("Policy loaded")
    set_gpu_mode(True)
    policy.cuda()
    return policy, env

# ----------------- Env construction -----------------

fair_maze_map = [
    [WW, WW, WW, WW, WW, WW],
    [WW, SP, EM, WW, O1, WW],
    [WW, EM, EM, EM, EM, WW],
    [WW, WW, EM, EM, D1, WW],
    [WW, O2, EM, D2, WW, WW],
    [WW, WW, WW, WW, WW, WW],
]

# ----------------- Low-level policy wrapper -----------------

def make_fair_goal_reaching_policy(env, policy):
    def fair_goal_reaching_policy_fn(obs, goal):
        reward_dim = env.reward_dim
        base_obs = obs[2:-reward_dim]  # remove (x,y) & goal_state bits
        goal_x, goal_y = goal
        goal_tuple = np.array([goal_x, goal_y])

        # normalize the norm of the relative goals to in-distribution values
        if np.linalg.norm(goal_tuple) > 1e-8:
            goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

        new_obs = np.concatenate([base_obs, goal_tuple], -1)
        action = policy.get_action(new_obs)[0]

        return action, (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])

    return fair_goal_reaching_policy_fn


# ----------------- Dataset collection -----------------

def collect_dataset(
    num_transitions=1_000_000,
    max_steps_per_episode=500,
    save_path="./datasets/fair_antmaze_dataset.npz",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Build env
    env = NormalizedBoxEnv(
        FairAntMaze(
            maze_map=fair_maze_map,
            maze_size_scaling=4.0,
            touch_threshold=2.0,
            maze_height=0.5,
            max_steps=max_steps_per_episode,
            expose_all_qpos=True,
        )
    )

    # 2. Load policy
    policy, _ = load_policy(POLICY_FILE)
    fair_goal_fn = make_fair_goal_reaching_policy(env, policy)
    # 3. Build navigation policy (high-level)
    script_options = [
            ["O1", "D1", "O2", "D2"], # switching policy
            ["O1", "D1"],             # Objective 1
            ["O2", "D2"],             # Objective 2
        ]

    # script = ["O1", "D1", "O2", "D2"]
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Reward Dimension: {env.reward_dim}")

    # 4. Buffers for transitions
    states      = []
    actions     = []
    rewards     = []
    next_states = []
    dones       = []
    init_states = []
    Raccs       = []
    next_Raccs  = []

    total_collected = 0
    episode_idx = 0

    # fair_goal_fn = make_fair_goal_reaching_policy(env, policy)
    # navigation_policy = env.create_fair_navigation_policy(
    #     low_level_policy_fn=fair_goal_fn,
    #     script_tokens=script,
    #     # obs_to_robot keeps default obs[:2] = (x, y)
    # )
    
    while total_collected < num_transitions:
        selected_script = script_options[np.random.randint(len(script_options))]
        
        navigation_policy = env.create_fair_navigation_policy(
            low_level_policy_fn=fair_goal_fn,
            script_tokens=selected_script,
        )
        
        obs = env.reset()
        done = False
        step_count = 0
        Racc = np.zeros(env.reward_dim, dtype=np.float32)
        init_obs = obs.copy()


        while (not done) and (step_count < max_steps_per_episode):
            action, _ = navigation_policy(obs)
            next_obs, reward, done, info = env.step(action)

            # Compute next accumulated return
            reward_vec = np.array(reward, dtype=np.float32)
            next_Racc = Racc + reward_vec

            # Store transition
            states.append(obs.astype(np.float32))
            actions.append(action.astype(np.float32))
            rewards.append(reward_vec)
            next_states.append(next_obs.astype(np.float32))
            dones.append(done)
            init_states.append(init_obs.astype(np.float32))
            Raccs.append(Racc.copy())
            next_Raccs.append(next_Racc.copy())

            # Update for next step
            obs = next_obs
            Racc = next_Racc
            step_count += 1
            total_collected += 1
            
            if total_collected % 1000 == 0:
                print(f"Collected {total_collected} transitions so far...")

        episode_idx += 1
        print(f"Episode {episode_idx} finished after {step_count} steps."
              f"Total collected: {total_collected}/{num_transitions}")
        print(f"  Final accumulated return: {Racc}")

    env.close()

    # 5. Convert to numpy arrays
    states      = np.stack(states, axis=0)
    actions     = np.stack(actions, axis=0)
    rewards     = np.stack(rewards, axis=0)
    next_states = np.stack(next_states, axis=0)
    dones       = np.array(dones, dtype=np.bool_)
    init_states = np.stack(init_states, axis=0)
    Raccs       = np.stack(Raccs, axis=0)
    next_Raccs  = np.stack(next_Raccs, axis=0)

    print("Final dataset shapes:")
    print(" states     :", states.shape)
    print(" actions    :", actions.shape)
    print(" rewards    :", rewards.shape)
    print(" next_states:", next_states.shape)
    print(" dones      :", dones.shape)
    print(" init_states:", init_states.shape)
    print(" Raccs      :", Raccs.shape)
    print(" next_Raccs :", next_Raccs.shape)

    # 6. Save to disk
    np.savez_compressed(
        save_path,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        initial_states=init_states,
        Raccs=Raccs,
        next_Raccs=next_Raccs,
    )
    print(f"Saved dataset to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_transitions", type=int, default=2000)
    parser.add_argument("--num_transitions", type=int, default=300_000)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="./datasets/fair_antmaze_300k.npz")
    args = parser.parse_args()
    
    
    collect_dataset(
        num_transitions=args.num_transitions,
        max_steps_per_episode=args.max_steps_per_episode,
        save_path=args.save_path,
    )
