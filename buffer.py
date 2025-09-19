
import numpy as np
import torch
from utils import *

def dataset_to_replaybuffer(
    dataset,
    env,
    device="cuda",
    # reward
    reward_mode="esr",
    utility=None,
    reward_index=0,
    # state
    one_hot_pass_idx=False,
    concat_acc_reward=False,
    ):
    """
    Convert dataset into replay buffer with decoded states.
    Uses env.decode() to decode state int -> features.
    Optionally concat accumulated reward vector Raccs to state.
    """
    N = len(dataset["states"])
    decoded_states, decoded_next_states, decoded_init_states = [], [], []

    num_dest = len(env.dest_coords) 
    
    for i in range(N):
        s = list(env.decode(dataset["states"][i]))
        ns = list(env.decode(dataset["next_states"][i]))
        init = list(env.decode(dataset["initial_states"][i]))

        if one_hot_pass_idx:
            s_pass_onehot = np.zeros(num_dest + 1, dtype=np.float32)
            ns_pass_onehot = np.zeros(num_dest + 1, dtype=np.float32)
            init_pass_onehot = np.zeros(num_dest + 1, dtype=np.float32)
            s_pass_onehot[s[3]] = 1.0
            ns_pass_onehot[ns[3]] = 1.0
            init_pass_onehot[init[3]] = 1.0

            s_feat = [s[0], s[1], s[2]] + s_pass_onehot.tolist()
            ns_feat = [ns[0], ns[1], ns[2]] + ns_pass_onehot.tolist()
            init_feat = [init[0], init[1], init[2]] + init_pass_onehot.tolist()
        else:
            s_feat, ns_feat, init_feat = s, ns, init

        if concat_acc_reward:
            racc = dataset["Raccs"][i].astype(np.float32)
            racc = racc.tolist()
            s_feat = s_feat + racc
            ns_feat = ns_feat + racc
            init_feat = init_feat + racc

        decoded_states.append(s_feat)
        decoded_next_states.append(ns_feat)
        decoded_init_states.append(init_feat)
        

    decoded_states = np.array(decoded_states, dtype=np.float32)
    decoded_next_states = np.array(decoded_next_states, dtype=np.float32)
    decoded_init_states = np.array(decoded_init_states, dtype=np.float32)

    # ---- reward: ESR or index ----
    if reward_mode == "esr":
        if utility is None:
            raise ValueError("reward_mode='esr'에서는 utility(Utility 인스턴스)가 필요합니다.")
        # ESR 계산은 원본 rewards / Raccs 사용
        print("esr...")
        rewards = esr_calculation(dataset['Raccs'], dataset['rewards'], utility)
        
    elif reward_mode == "scalar_index":
        rewards = dataset["rewards"][:, reward_index].astype(np.float32).reshape(-1, 1)
    else:
        raise ValueError("reward_mode must be 'esr' or 'scalar_index'")        

    # ---- ReplayBuffer 생성 ----
    state_dim = decoded_states.shape[1]
    buffer = ReplayBuffer(state_dim=state_dim, capacity=N, device=device)
    print("state shape:", decoded_states.shape)
    
    
     # 데이터 삽입
    for i in range(N):
        buffer.push(
            state=decoded_states[i],
            action=dataset['actions'][i],
            reward=rewards[i],   # 보상 스케일링 (선택)
            next_state=decoded_next_states[i],
            done=dataset.get("dones", np.zeros(N))[i],   # 없으면 0으로
            t=dataset['timesteps'][i],
            next_t=dataset['timesteps'][i] + 1,
            init_state=decoded_init_states[i],
        )
        
    return buffer


class ReplayBuffer:
    def __init__(self, state_dim, capacity, device="cuda"):
        self.capacity = capacity
        self.device = device

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.init_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)   # store discrete actions
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.timesteps = np.zeros((capacity, 1), dtype=np.int64)
        self.next_timesteps = np.zeros((capacity, 1), dtype=np.int64)


        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, t, next_t, init_state):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action   # int
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.init_states[self.ptr] = init_state
        self.timesteps[self.ptr] = t
        self.next_timesteps[self.ptr] = next_t

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.states[idxs], device=self.device)
        next_states = torch.tensor(self.next_states[idxs], device=self.device)
        init_states = torch.tensor(self.init_states[idxs], device=self.device)
        actions = torch.tensor(self.actions[idxs], device=self.device).squeeze(-1)  # [B]
        rewards = torch.tensor(self.rewards[idxs], device=self.device)
        dones = torch.tensor(self.dones[idxs], device=self.device)

        masks = 1.0 - dones

        timesteps = torch.tensor(self.timesteps[idxs], device=self.device).long()
        next_timesteps = torch.tensor(self.next_timesteps[idxs], device=self.device).long()

        return {
            "states": states,
            "next_states": next_states,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,                # 1 - done
            "timesteps": timesteps,        # [B,1] long
            "next_timesteps": next_timesteps,
            "init_states": init_states,    # [B, state_dim] float
        }

    def __len__(self):
        return self.size

