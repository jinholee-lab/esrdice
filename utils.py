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
        # loc_coords = [[0,0],[3,2]]
        # dest_coords = [[0,4],[3,3]]
        # loc_coords = [[1,1],[8,7]]
        # dest_coords = [[1,4],[8,8]]
        # loc_coords = [[0,2],[9,7]]
        # dest_coords = [[0,4],[9,8]]
        loc_coords = [[0,2], [9,7]]
        dest_coords = [[0,0], [9,9]]
    elif num_locs == 3:
        # loc_coords = [[0,0],[0,5],[3,2]]
        # dest_coords = [[0,4],[5,0],[3,3]]
        loc_coords = [[0,0],[4,2], [9,1]]
        dest_coords = [[0,4],[5,3], [9,3]]
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

EPS = 1e-12

class Utility:
    """
    Utility function for multi-objective RL
    kind: "linear" | "log" | "piecewise_log"
      - linear       : u(R)= R_i
      - log          : u(R)= log(R_i)
      - piecewise_log: u(R)= u_piecewise(R_i)
      - alpha_fairness: u(R)= u_alpha_fairness(R_i) 
    """
    def __init__(self, kind="linear", alpha=0.5):
        assert kind in ("linear", "log", "piecewise_log", "alpha_fairness"), f"Unknown utility kind: {kind}"
        self.kind    = kind
        self.alpha   = alpha

    def _log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x+EPS)

    def _piecewise_log(self, x: np.ndarray) -> np.ndarray:
        """
        piecewise utility:
        - x >= 1: log(x)
        - x <  1: -0.5 * (x - 2)^2 + 0.5
        """
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        mask = (x >= 1)
        out[mask] = np.log(x[mask])   # log region
        out[~mask] = -0.5 * (x[~mask] - 2.0)**2 + 0.5  # quadratic region
        return out
    
    def _frac(self, x: np.ndarray) -> np.ndarray:
        return ((x + EPS)**(1 - self.alpha) - 1) / (1 - self.alpha)
    
    def _piecewise_frac(self, x: np.ndarray) -> np.ndarray:
        """
        piecewise utility:
        - x >= 1: (x^{1-alpha} - 1) / (1 - alpha)
        - x <  1 : e^{-alpha}(x-1) + (1-e^{-alpha})(-0.5*(x-2)^2 + 0.5)
        """
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        mask = (x >= 1)
        out[mask] = (x[mask]**(1 - self.alpha) - 1) / (1 - self.alpha)   # fractional region

        s = float(np.exp((-self.alpha)))  # e^{-α}
        q_lin  = x[~mask] - 1.0
        q_quad = -0.5*(x[~mask] - 2.0)**2 + 0.5
        out[~mask] = s * q_lin + (1.0 - s) * q_quad

        return out
    
    def _piecewise_frac(self, x: np.ndarray) -> np.ndarray:
        """
        piecewise utility:
        - x >= 1: (x^{1-alpha} - 1) / (1 - alpha)
        - x <  1 : e^{-alpha}(x-1) + (1-e^{-alpha})(-0.5*(x-2)^2 + 0.5)
        """
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        mask = (x >= 1)
        out[mask] = (x[mask]**(1 - self.alpha)) / (1 - self.alpha)   # fractional region

        s = float(np.exp((-self.alpha)))  # e^{-α}
        q_lin  = x[~mask]
        q_quad = -0.5*(x[~mask] - 2.0)**2 + 0.5
        out[~mask] = s * q_lin + (1.0 - s) * q_quad

        return out

    def _alpha_fairness(self, x: np.ndarray) -> np.ndarray:
        if self.alpha == 1.0:
            return self._piecewise_log(x)
            # return self._log(x)
        else:
            return self._piecewise_frac(x)
            # return self._frac(x)

    def _transform(self, R: np.ndarray) -> np.ndarray:
        if self.kind == "linear":
            return R
        elif self.kind == "log":
            return self._log(R)
        elif self.kind == "piecewise_log":
            return self._piecewise_log(R)
        elif self.kind == "alpha_fairness":
            return self._alpha_fairness(R)
        else:
            raise NotImplementedError

    def __call__(self, R: np.ndarray, keep_dims: bool = False) -> np.ndarray:
        R = np.asarray(R, dtype=np.float32)  # [N,D]
        X = self._transform(R)          # [N,D]
        
        if keep_dims:
            return X               # [N,D]
        else:
            return X.sum(axis=-1)  # [N]