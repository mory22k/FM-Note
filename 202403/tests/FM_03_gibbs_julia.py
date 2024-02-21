import numpy as np
from pathlib import Path
from typing import Optional
from julia import Main

current_dir = Path(__file__).resolve().parent
julia_file = current_dir / "FM_03_gibbs_julia.jl"
Main.include(str(julia_file))

# Factorization Machine

class FactorizationMachines:
    def __init__(self,
        num_features: int,
        num_factors:  int,
        sigma_b_init: float=0.,
        sigma_w_init: float=1.,
        sigma_v_init: float=1.,
        seed: Optional[int]=None
    ) -> None:
        self.rng = np.random.default_rng(seed)
        b = self.rng.normal(0, sigma_b_init)
        w = self.rng.normal(0, sigma_w_init, num_features)
        v = self.rng.normal(0, sigma_v_init, (num_features, num_factors))
        self.params = {'b': b, 'w': w, 'v': v}

    def predict(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            x = x.reshape(1, -1) # x: (d, n)
        b = self.params['b']     # b: (1)
        w = self.params['w']     # w: (d)
        v = self.params['v']     # v: (d, k)

        bias   = b
            # (1)
        linear = x[:, :] @ w[:]
            # (D, N) @ (N) = (D)
        inter  = 0.5 * np.sum((x[:, :] @ v[:, :]) ** 2 - (x[:, :] ** 2) @ (v[:, :] ** 2), axis=1)
            # (D, K) -> (D)

        result = bias + linear + inter
            # (D)

        if result.shape[0] == 1:
            return float(result[0])
        return result

# test
N = 16
K = 8
D = 128

seed = 0
rng  = np.random.default_rng(seed)
fm   = FactorizationMachines(N, K, seed=seed)

Q      = rng.uniform(-1., 1., (N, N))
x_data = rng.choice((0, 1), size=(D, N))
y_data = np.einsum('dn,nm,dm->d', x_data, Q, x_data)

model_params = fm.params

latent_params = {
    'mu_w': 0.,
    'mu_v': np.zeros(K, dtype=float),
    'sigma2_w': 1.,
    'sigma2_v': np.ones(K, dtype=float)
}

fixed_params = {
    'mu_b': 0.,
    'sigma2_b': 1.,
    'm_w': 0.,
    'm_v': 0.,
    'l_w': 1.,
    'l_v': 1.,
    'a_w': 1.,
    'a_v': 1.,
    'b_w': 1.,
    'b_v': 1.,
    'a_noise': 1.,
    'b_noise': 1.,
}

_, loss_hist = Main.train_fm_gibbs(
    model_params,
    latent_params,
    fixed_params,
    x_data.astype(float),
    y_data,
    fm.predict(x_data),
    10000,
)
