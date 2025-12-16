# scripts/run_fair_antmaze.py
import os
import imageio
import numpy as np
import torch
from rlkit.torch.pytorch_util import set_gpu_mode


from fair_maze_env import FairAntMaze, WW, EM, SP, O1, D1, O2, D2
from wrappers import NormalizedBoxEnv

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
        goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

        new_obs = np.concatenate([base_obs, goal_tuple], -1)
        return policy.get_action(new_obs)[0], (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])

    return fair_goal_reaching_policy_fn

# ----------------- Main rollout function -----------------

def run_episode(env, navigation_policy, max_steps=1000, save_path="./vis/test.gif", fps=20):
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = np.zeros(env.reward_dim)
    frames = []

    while not done and step_count < max_steps:
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        action, waypoint = navigation_policy(obs)
        print("action norm:", np.linalg.norm(action), "max abs:", np.max(np.abs(action)))

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        print(f"Step {step_count}: XY Coordinates: {env.get_xy()} Reward: {reward}, Return: {total_reward}")

        if np.any(reward > 0):
            print(f"Step {step_count}: REWARD! {reward}, Total: {total_reward}")
            print(f"  New goal states: {info['goal_states']}")

    print(f"Episode finished after {step_count} steps. Total Reward: {total_reward}")

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"Saved GIF to {save_path}")

    env.close()

def main():
    
    env = NormalizedBoxEnv(
            FairAntMaze(
                maze_map=fair_maze_map,
                maze_size_scaling=4.0,
                touch_threshold=2.0,
                maze_height=0.5,
                max_steps=1000,
                expose_all_qpos=True,
                )
            )
    policy, _ = load_policy(POLICY_FILE)
    
    script = ["O1", "D1", "O2", "D2"]
    fair_goal_fn = make_fair_goal_reaching_policy(env, policy)

    navigation_policy = env.create_fair_navigation_policy(
        low_level_policy_fn=fair_goal_fn,
        script_tokens=script,
        # obs_to_robot keeps default obs[:2], which is still (x,y)
    )

    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Reward Dimension: {env.reward_dim}")

    run_episode(env, navigation_policy, max_steps=1000, save_path="./vis/test.gif", fps=20)

if __name__ == "__main__":
    main()
