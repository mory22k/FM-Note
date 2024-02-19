import numpy as np
from typing import Optional

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

def calc_q_init(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    # x: (D, N)
    # v: (N, K)
    return x[:, :] @ v[:, :] # (D, K)

def calc_dq(x: np.ndarray, v_ik_new: float, v: np.ndarray, i: int, k: int) -> np.ndarray:
    # v_ik_new: float
    # v: (N, K)
    # x: (D, N)
    return (v_ik_new - v[i, k]) * x[:, i] # (D)

def calc_df(h: np.ndarray, p_new: float, p_old: float) -> np.ndarray:
    # h: (D)
    return (p_new - p_old) * h # (D)

def calc_h_b(x) -> float:
    return np.ones(x.shape[0])

def calc_h_w(x: np.ndarray, i: int) -> np.ndarray:
    # x: (D, N)
    return x[:, i]

def calc_h_v_fast(x: np.ndarray, v: np.ndarray, q: np.ndarray, i: int, k: int) -> np.ndarray:
    # x: (D, N)
    # v: (N, K)
    # q: (D, K)
    return x[:, i] * (q[:, k] - x[:, i] * v[i, k]) # (D)

def calc_g(f: np.ndarray, h: np.ndarray, p: float) -> np.ndarray:
    g = f - h * p
    return g

def sample_param_lstsq(y: np.ndarray, h: np.ndarray, g: np.ndarray, lamb: float=1e-8) -> float:
    # h: (D)
    # g: (D)
    x_theta = h
    y_theta = y - g
    return np.sum(x_theta * y_theta) / (np.sum(x_theta ** 2) + lamb)

def train_fm_als(
    init_params: dict,
    x_data: np.ndarray,
    y_data: np.ndarray,
    f_init: np.ndarray,
    num_iter: int,
) -> dict:
    # get indices
    N = x_data.shape[1]
    K = init_params['v'].shape[1]

    # get initial parameter
    params = init_params

    # precalculate
    f = f_init
    q = calc_q_init(x_data, params['v'])

    # main loop
    for iter in range(num_iter):
        # sample b
        h_b   = calc_h_b(x_data)
        g_b   = calc_g(f, h_b, params['b'])
        b_new = sample_param_lstsq(y_data, h_b, g_b)
        f     = f + calc_df(h_b, b_new, params['b'])
        params['b'] = b_new

        # sample w
        for i in range(N):
            h_w   = calc_h_w(x_data, i)
            g_w   = calc_g(f, h_w, params['w'][i])
            w_i_new = sample_param_lstsq(y_data, h_w, g_w)
            f     = f + calc_df(h_w, w_i_new, params['w'][i])
            params['w'][i] = w_i_new

        # sample v
        for i in range(N):
            for k in range(K):
                h_v      = calc_h_v_fast(x_data, params['v'], q, i, k)
                g_v      = calc_g(f, h_v, params['v'][i, k])
                v_ik_new = sample_param_lstsq(y_data, h_v, g_v)
                f        = f      + calc_df(h_v, v_ik_new, params['v'][i, k])
                q[:,k]   = q[:,k] + calc_dq(x_data, v_ik_new, params['v'], i, k)
                params['v'][i, k] = v_ik_new

        if iter % 10 == 0:
            print(f'iter: {iter}, loss: {np.mean((y_data - f) ** 2)}')

    return params

# test
N = 16
K = 8
D = 128

seed = 0
rng  = np.random.default_rng(seed)
fm   = FactorizationMachines(N, K, seed=seed)
x    = rng.choice((0, 1), size=(D, N))

Q    = rng.uniform(-1., 1., (N, N))
y    = np.einsum('dn,nm,dm->d', x, Q, x)

_ = train_fm_als(
    fm.params, x, y, fm.predict(x), 1000
)
