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

def calc_q_init(
    x: np.ndarray,
    v: np.ndarray
) -> np.ndarray:
    # x: (D, N)
    # v: (N, K)
    return x[:, :] @ v[:, :] # (D, K)

def calc_dq(
    i: int,
    x: np.ndarray,
    v_ik_new: float,
    v_ik_old: np.ndarray,
) -> np.ndarray:
    # v_ik_new: float
    # v: (N, K)
    # x: (D, N)
    return (v_ik_new - v_ik_old) * x[:, i] # (D)

def calc_df(
    x_theta: np.ndarray,
    param_new: float,
    param_old: float,
):
    return (param_new - param_old) * x_theta

def calc_xy_b(
    f: np.ndarray,
    b: float,
    x_data: np.ndarray,
    y_data: np.ndarray,
):
    # x_data: (D, N)
    # y_data: (D)
    x_b = np.ones(x_data.shape[0])
    y_b = y_data - (f - b * x_b)
    return x_b, y_b

def calc_xy_w(
    f: np.ndarray,
    w: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    i: int
):
    # x_data: (D, N)
    # y_data: (D)
    x_w = x_data[:, i]
    y_w = y_data - (f - x_w * w[i])
    return x_w, y_w

def calc_xy_v(
    f: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    i: int,
    k: int
):
    # x_data: (D, N)
    # y_data: (D)
    x_v = x_data[:, i] * (q[:, k] - x_data[:, i] * v[i, k])
    y_v = y_data - (f - x_v * v[i, k])
    return x_v, y_v

def sample_param_lstsq(
    x_theta: np.ndarray,
    y_theta: np.ndarray,
    lamb: float=1e-8
) -> float:
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
        x_b, y_b = calc_xy_b(f, params['b'], x_data, y_data)
        b_new    = sample_param_lstsq(x_b, y_b)
        f        = f + calc_df(x_b, b_new, params['b'])
        params['b'] = b_new

        # sample w
        for i in range(N):
            x_w, y_w = calc_xy_w(f, params['w'], x_data, y_data, i)
            w_i_new  = sample_param_lstsq(x_w, y_w)
            f        = f + calc_df(x_w, w_i_new, params['w'][i])
            params['w'][i] = w_i_new

        # sample v
        # for i in range(N):
            for k in range(K):
                x_v, y_v = calc_xy_v(f, q, params['v'], x_data, y_data, i, k)
                v_ik_new = sample_param_lstsq(x_v, y_v)
                f        = f      + calc_df(x_v, v_ik_new, params['v'][i, k])
                q[:,k]   = q[:,k] + calc_dq(i, x_data, v_ik_new, params['v'][i, k])
                params['v'][i, k] = v_ik_new

        if iter % 50 == 0:
            print(f'iter: {iter}, loss: {np.sum((y_data - f) ** 2) / N}')

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
