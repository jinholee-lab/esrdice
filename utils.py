import numpy as np


def get_setting(size, num_locs):
    """
    To store environment settings

    Parameters
    ----------
    size : int
        size of the grid world in N x N
    num_locs : int
        number of location destination pairs
    """
    if num_locs == 2:
        loc_coords = [[0,0],[3,2]]
        dest_coords = [[0,4],[3,3]]
    elif num_locs == 3:
        loc_coords = [[0,0],[0,5],[3,2]]
        dest_coords = [[0,4],[5,0],[3,3]]
    elif num_locs == 4:
        loc_coords = [[0,0], [0,5], [3,2], [9,0]]
        dest_coords = [[0,4], [5,0], [3,3], [0,9]]
    elif num_locs == 5:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[4,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[8,9]]
    else:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[8,9],[6,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[4,7],[8,3]]
    return size, loc_coords, dest_coords

def process_states(states, num_episodes=600, horizon=100):
    """
    states: shape (num_episodes*horizon,)
    num_episodes: number of episodes (e.g., 600)
    horizon: fixed horizon length (e.g., 100)
    
    return:
        reshaped_states: (num_episodes, horizon)
        initial_states: (num_episodes, horizon)
        initial_states_flat: (num_episodes*horizon,)
    """
    # (1) reshape to (num_episodes, horizon)
    reshaped_states = states.reshape(num_episodes, horizon)
    
    # (2) 각 episode의 첫 state를 initial state로 채움
    initial_states = np.repeat(reshaped_states[:, [0]], horizon, axis=1)
    
    # (3) 다시 flatten
    initial_states_flat = initial_states.reshape(-1)
    
    return reshaped_states, initial_states, initial_states_flat


# 사용 예시
# states = dataset['states']
# reshaped_states, initial_states, initial_states_flat = process_states(states)

# print("reshaped_states.shape:", reshaped_states.shape)  # (600, 100)
# print("initial_states.shape:", initial_states.shape)    # (600, 100)
# print("initial_states_flat.shape:", initial_states_flat.shape)  # (60000,)

def recompute_raccs(dataset, horizon):
    """
    dataset: dict with keys 'rewards' [N, D], 'timesteps' [N]
    horizon: fixed horizon length
    returns updated Raccs [N, D]
    """
    N, D = dataset["rewards"].shape
    Raccs = np.zeros((N, D), dtype=np.float32)

    # 각 trajectory별로 timesteps를 0~H-1 순서대로 누적
    # dataset은 여러 trajectory가 concat된 상태라고 가정
    for start in range(0, N, horizon):
        acc = np.zeros(D, dtype=np.float32)
        for t in range(horizon):
            idx = start + t
            Raccs[idx] = acc
            acc += dataset["rewards"][idx]
    return Raccs


def normalize_dataset(dataset, keys, method="zscore"):
    """
    dataset: dict 형태 (numpy array 포함)
    keys: list of str, normalize할 key 이름들
    method: str, "zscore" 또는 "minmax"
    """
    new_dataset = dataset.copy()
    stat_dict = {}
    for key in keys:
        if key not in dataset:
            raise KeyError(f"{key} is not in dataset")
        
        arr = dataset[key].astype(np.float32)
        
        if method == "zscore":
            mean = arr.mean()
            std = arr.std()
            print(f"mean.shape{mean}")
            if std > 0:
                new_dataset[key] = (arr - mean) / std
            else:
                new_dataset[key] = arr  # std=0이면 그대로 둠
            
            stat_dict[key] = {"mean" : mean,
                              "std" : std}
                
        elif method == "minmax":
            min_val = arr.min()
            max_val = arr.max()
            if max_val > min_val:
                new_dataset[key] = (arr - min_val) / (max_val - min_val)
            else:
                new_dataset[key] = arr  # 값이 모두 같으면 그대로 둠
            stat_dict[key] = {"max" : max_val,
                                "min" : min_val}
                
        else:
            raise ValueError("method must be 'zscore' or 'minmax'")
    
    return new_dataset, stat_dict


class Utility:
    """
    u(R)를 스칼라로 반환.
    kind: "linear" | "log"
      - linear: u(R)=w^T R
      - log   : u(R)=sum_i w_i * log(R_i + shift)
    weights가 None이면 입력 차원에 맞춰 균등가중치로 자동 설정.
    """
    def __init__(self, kind="linear", weights=None, shift=0.0, eps=1e-6):
        assert kind in ("linear", "log")
        self.kind    = kind
        self.weights = np.array([1.0, 1.0]) if weights is None else np.asarray(weights, np.float32)
        self.shift   = float(shift)
        self.eps     = eps

    def _ensure_weights(self, R):
        if self.weights is None:
            D = R.shape[-1]
            self.weights = np.ones(D, dtype=np.float32) / float(D)

    def __call__(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float32)  # [N,D] or [D]
        self._ensure_weights(R) 
        if self.kind == "linear":
            return (R * self.weights).sum(axis=-1)
        else:  # "log"
            return (np.log(R + self.shift + self.eps)).sum(axis=-1)


def esr_calculation(Raccs: np.ndarray, rewards: np.ndarray, utility: Utility) -> np.ndarray:
    """
    ESR step reward (scalar):
      r_esr(t) = u(Racc_t + r_t) - u(Racc_t)
    Args:
      Raccs: [N, D]
      rewards: [N, D]
      utility: Utility
    Returns:
      esr: [N, 1] float32
    """
    Raccs  = np.asarray(Raccs,  dtype=np.float32)
    rewards= np.asarray(rewards, dtype=np.float32)
    delta  = utility(Raccs + rewards) - utility(Raccs)      # [N]
    return delta.astype(np.float32)[:, None]
