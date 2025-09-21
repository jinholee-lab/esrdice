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

class Utility:
    """
    u(R)를 스칼라로 반환.
    kind: "linear" | "log" | "piecewise_log"
      - linear       : u(R)=w^T R
      - log          : u(R)=sum_i w_i * log(R_i + shift)
      - piecewise_log: u(R)=sum_i w_i * u_piecewise(R_i)
                       (log(x)와 음수 허용 구간까지 매끄럽게 연결한 함수)
    weights가 None이면 입력 차원에 맞춰 균등가중치로 자동 설정.
    """
    def __init__(self, kind="linear", weights=None, shift=0.0, eps=1e-6):
        assert kind in ("linear", "log", "piecewise_log")
        self.kind    = kind
        self.weights = np.array([1.0, 1.0]) if weights is None else np.asarray(weights, np.float32)
        self.shift   = float(shift)
        self.eps     = eps

    def _ensure_weights(self, R):
        if self.weights is None:
            D = R.shape[-1]
            self.weights = np.ones(D, dtype=np.float32) / float(D)

    def _piecewise_log(self, x: np.ndarray) -> np.ndarray:
        """
        이미지에서 설명된 piecewise utility:
        - x >= 1: log(x)
        - x <  1: -0.5 * (x - 2)^2 + 0.5
        """
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        mask = (x >= 1)
        out[mask] = np.log(x[mask] + self.eps)   # log 구간
        out[~mask] = -0.5 * (x[~mask] - 2.0)**2 + 0.5  # 이차식 구간
        return out

    def __call__(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float32)  # [N,D] or [D]
        self._ensure_weights(R) 

        if self.kind == "linear":
            return (R * self.weights).sum(axis=-1)
        elif self.kind == "log":
            return (np.log(R + self.shift + self.eps) * self.weights).sum(axis=-1)
        else:  # "piecewise_log"
            return (self._piecewise_log(R) * self.weights).sum(axis=-1)
