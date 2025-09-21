import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=None, device="cpu"):
        """
        capacity: 최대 저장 용량 (None이면 dataset 크기와 동일)
        device: 샘플 반환 시 올릴 장치
        """
        self.capacity = capacity
        self.device = device
        self.state_dim = None
        self.action_dim = None
        self.reset()

    def reset(self):
        self.buffer = {}
        self.size = 0

    def load_dataset(self, dataset):
        """
        dataset(dict) → replay buffer로 로드
        필수 키: states, actions, rewards, next_states, dones
        선택 키: initial_states, Raccs, next_Raccs 등도 같이 저장 가능
        """
        self.buffer = {}
        for k, v in dataset.items():
            self.buffer[k] = np.array(v)
        self.size = len(self.buffer["states"])
        if self.capacity is None:
            self.capacity = self.size
        print(f"[ReplayBuffer] Loaded dataset with {self.size} transitions.")

    def sample(self, batch_size):
        """
        batch_size 크기의 transition 샘플링
        return: dict of torch tensors
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {}
        for k, v in self.buffer.items():
            if not (isinstance(v, np.ndarray) and len(v) == self.size):
                continue
            if k == "actions" or k == "timesteps" or k == "next_timesteps":
                batch[k] = torch.as_tensor(v[idxs], device=self.device, dtype=torch.long)
            else:
                batch[k] = torch.as_tensor(v[idxs], device=self.device, dtype=torch.float32)
        return batch

    def all_data(self):
        """
        전체 데이터를 torch tensor로 반환
        (ex. evaluation 시 필요할 때)
        """
        out = {}
        for k, v in self.buffer.items():
            if k == "actions":
                out[k] = torch.as_tensor(v, device=self.device, dtype=torch.long)
            else:
                out[k] = torch.as_tensor(v, device=self.device, dtype=torch.float32)
        return out