import numpy as np
import torch
from utility import Utility
class ReplayBuffer:
    def __init__(self, capacity=None, device="cpu",
                 # params for sample augmentation
                 utility: Utility = None, 
                 horizon: int = None,
                 keep_dims: bool = False,
                 reward_dim: int = None,
                 aug_ratio: float = 0.5,
                 use_augmentation: bool = True
                 ):
        """
        capacity: 최대 저장 용량 (None이면 dataset 크기와 동일)
        device: 샘플 반환 시 올릴 장치
        """
        self.capacity = capacity
        self.device = device
        
        self.utility = utility
        self.horizon = horizon
        self.keep_dims = keep_dims
        self.reward_dim = reward_dim
        self.aug_ratio = aug_ratio
        self.use_augmentation = use_augmentation
        # Calculate u(0) once and store it (on device)
        u0 = self.utility(
                        torch.zeros(1,self.reward_dim, dtype=torch.float32),
                        keep_dims=self.keep_dims,
                        ).squeeze(0)
        self.u0 = torch.as_tensor(u0, device=self.device, dtype=torch.float32)
        
        
        self.state_dim = None
        self.action_dim = None
        self.reset()

    def reset(self):
        self.buffer = {}
        self.size = 0
        
        self.unique_Raccs = None
        self.num_unique_Raccs = 0

    def load_dataset(self, dataset):
        """
        dataset(dict) → replay buffer
        Keys: states, actions, rewards, next_states, dones, initial_states, Raccs, next_Raccs 
        """
        self.buffer = {}
        
        self.size = len(dataset["states"])
                                
        if self.capacity is None:
            self.capacity = self.size
        
        print(f"[ReplayBuffer] Loading and transferring {self.size} transitions to {self.device}...")
        
        for k, v in dataset.items():
            if k in ['actions', 'timesteps', 'next_timesteps']:
                dtype = torch.long
            else:
                dtype = torch.float32
            
            self.buffer[k] = torch.as_tensor(v, device=self.device, dtype=dtype)

            
        if "Raccs" in self.buffer:
            Raccs = self.buffer["Raccs"]
            self.unique_Raccs = torch.unique(Raccs, dim=0)
            self.num_unique_Raccs = self.unique_Raccs.shape[0]
        else:
            raise ValueError("Dataset must contain 'Raccs' key for reward augmentation.")
            
        print(f"[ReplayBuffer] Loaded dataset with {self.size} transitions. (to {self.device})")
        
    def _scalarize(self, Raccs, rewards, timesteps):
        u_Racc = self.utility(Raccs, keep_dims=self.keep_dims)
        u_Racc_next = self.utility(Raccs + rewards, keep_dims=self.keep_dims)
        scalarized_rewards = u_Racc_next - u_Racc
        
        is_first_step = (timesteps == 0).float()
        if self.keep_dims:
            scalarized_rewards= scalarized_rewards + self.u0.unsqueeze(0) * is_first_step.unsqueeze(-1)
        else:
            scalarized_rewards= scalarized_rewards + self.u0 * is_first_step
        return scalarized_rewards

    def sample(self, batch_size):
        """
        batch_size sampling
        return: dict of torch tensors
        """
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        states = self.buffer["states"][idxs]
        actions = self.buffer["actions"][idxs]
        original_rewards = self.buffer["rewards"][idxs]
        next_states = self.buffer["next_states"][idxs]
        initial_states = self.buffer["initial_states"][idxs]        
        timesteps = self.buffer["timesteps"][idxs]
        next_timesteps = self.buffer["next_timesteps"][idxs]
        
        # original Raccs and next_Raccs
        Raccs_original = self.buffer["Raccs"][idxs]
        Raccs_next_original = self.buffer["next_Raccs"][idxs]
        
        # Augmentation
        #   Raccs and next_Raccs
        if self.use_augmentation:
            Raccs_indices = torch.randint(0, self.num_unique_Raccs, (batch_size,), device=self.device) 
            Raccs_aug = self.unique_Raccs[Raccs_indices]
            Raccs_next_aug = Raccs_aug + original_rewards
            
            # mask to decide whether to augment or not
            augment_mask = (torch.rand(batch_size, device=self.device) < self.aug_ratio).unsqueeze(-1)
            Raccs_final = torch.where(augment_mask, Raccs_aug, Raccs_original)
            Raccs_next_final = torch.where(augment_mask, Raccs_next_aug, Raccs_next_original)
        else:
            Raccs_final = Raccs_original
            Raccs_next_final = Raccs_next_original
        
        #   timesteps and next_timesteps      
        # timesteps_aug = torch.randint(0, self.horizon, (batch_size,), device=self.device)
        # next_timesteps_aug = timesteps_aug + 1
        
        #   Concatenate augmented Raccs to states
        states_aug = torch.cat([states, Raccs_final], dim=-1)
        next_states_aug = torch.cat([next_states, Raccs_next_final], dim=-1)
        initial_Raccs = torch.zeros_like(Raccs_final, device=self.device)
        initial_states_aug = torch.cat([initial_states, initial_Raccs], dim=-1)
        
        # Scalarization
        scalarized_rewards = self._scalarize(Raccs_final, original_rewards, timesteps)
        
        batch = {
            "states": states_aug,
            "actions": actions,
            "rewards": scalarized_rewards,
            "next_states": next_states_aug,
            "initial_states": initial_states_aug,
            "timesteps": timesteps,
            "next_timesteps": next_timesteps,
        }
        
        return batch

    def all_data(self):
        out = {}
        for k, v in self.buffer.items():
            if k == "actions":
                out[k] = torch.as_tensor(v, device=self.device, dtype=torch.long)
            else:
                out[k] = torch.as_tensor(v, device=self.device, dtype=torch.float32)
        return out