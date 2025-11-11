import torch

EPS = 1e-12 # Epsilon for numerical stability

class Utility:
    """
    PyTorch-native Utility function for multi-objective RL
    Accepts and returns PyTorch Tensors.
    
    kind: "linear" | "log" | "piecewise_log" | "alpha_fairness"
    """
    def __init__(self, kind="linear", alpha=0.5):
        assert kind in ("linear", "log", "piecewise_log", "alpha_fairness"), f"Unknown utility kind: {kind}"
        self.kind  = kind
        self.alpha = alpha

    def _log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + EPS)

    def _piecewise_log(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch version of piecewise utility:
        - x >= 1: log(x)
        - x <  1: -0.5 * (x - 2)^2 + 0.5
        """
        mask = (x >= 1.0)
        
        log_vals = torch.log(x) # log(x) for x >= 1
        
        # -0.5 * (x - 2)^2 + 0.5 for x < 1
        quad_vals = -0.5 * torch.pow(x - 2.0, 2) + 0.5
        
        # Use torch.where to combine
        return torch.where(mask, log_vals, quad_vals)
    
    def _frac(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.pow(x + EPS, 1.0 - self.alpha) - 1.0) / (1.0 - self.alpha)
    
    def _piecewise_frac(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch version of piecewise fractional utility.
        (Using the first definition in your file)
        - x >= 1: (x^{1-alpha} - 1) / (1 - alpha)
        - x <  1 : e^{-alpha}(x-1) + (1-e^{-alpha})(-0.5*(x-2)^2 + 0.5)
        """
        mask = (x >= 1.0)
        
        # Fractional region (x >= 1)
        frac_vals = (torch.pow(x, 1.0 - self.alpha) - 1.0) / (1.0 - self.alpha)

        # Quadratic/Linear region (x < 1)
        s = torch.exp(torch.tensor(-self.alpha, device=x.device, dtype=x.dtype)) # e^{-α}
        
        q_lin  = x - 1.0
        q_quad = -0.5 * torch.pow(x - 2.0, 2) + 0.5
        
        linquad_vals = s * q_lin + (1.0 - s) * q_quad
        
        return torch.where(mask, frac_vals, linquad_vals)

    def _alpha_fairness(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 1.0:
            return self._piecewise_log(x)
        else:
            return self._piecewise_frac(x)
            # Note: Your original file had two different definitions for
            # _piecewise_frac. I used the first, more complex one.

    def _transform(self, R: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return R
        elif self.kind == "log":
            return self.log(R)
        elif self.kind == "piecewise_log":
            return self._piecewise_log(R)
        elif self.kind == "alpha_fairness":
            return self._alpha_fairness(R)
        else:
            raise NotImplementedError

    def __call__(self,
                 R: torch.Tensor,
                 keep_dims: bool = False,
                 ) -> torch.Tensor:
        
        # No np.asarray needed. R is already a tensor.
        X = self._transform(R) # [N,D]
        
        if keep_dims:
            return X            # [N,D]
        else:
            return X.sum(axis=-1) # [N]