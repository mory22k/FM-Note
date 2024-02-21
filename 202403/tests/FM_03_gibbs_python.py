import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

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

# update rules

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

# samplers
def sample_from_inverse_gamma(
    a: float,
    b: float,
    seed: Optional[int]=None
):
    rng = np.random.default_rng(seed)
    return 1 / rng.gamma(a, 1 / b)

def sample_posterior_sigma2_noise(
    x_theta: np.ndarray,
    y_theta: np.ndarray,
    theta: float,
    a_noise: float,
    b_noise: float,
    seed: Optional[int]=None
):
    # x_theta: (D)
    # y_theta: (D)
    D = x_theta.shape[0]
    a_noise_post = a_noise + D / 2
    b_noise_post = b_noise + 0.5 * np.sum( (y_theta - x_theta * theta)**2 )
    return sample_from_inverse_gamma(a_noise_post, b_noise_post, seed)

def sample_posterior_theta(
    x_theta: np.ndarray,
    y_theta: np.ndarray,
    sigma2_noise: float,
    mu_theta: float,
    sigma2_theta: float,
    seed: Optional[int]=None
):
    rng = np.random.default_rng(seed)
    sigma2_theta_post = 1 / ( np.sum(x_theta**2) / sigma2_noise + 1 / sigma2_theta )
    mu_theta_post = sigma2_theta_post * ( np.sum(x_theta * y_theta) / sigma2_noise + mu_theta / sigma2_theta )
    return rng.normal(mu_theta_post, np.sqrt(sigma2_theta_post))

def sample_posterior_mu_theta_and_sigma2_theta(
    theta: np.ndarray[float],
    sigma2_theta: float,
    m_theta: float,
    l_theta: float,
    a_theta: float,
    b_theta: float,
    seed: Optional[int]=None
):
    # theta: (N)
    rng = np.random.default_rng(seed)

    N = theta.shape[0]

    l_theta_post = l_theta + N
    mu_theta_post = (np.sum(theta) + l_theta * m_theta) / l_theta_post
    mu_theta_new  = rng.normal(mu_theta_post, np.sqrt(sigma2_theta / l_theta_post))

    mu_theta = mu_theta_new

    a_theta_post = a_theta + (N + 1) / 2
    b_theta_post = b_theta + 0.5 * (np.sum((theta - mu_theta)**2) + l_theta * (mu_theta - m_theta)**2)

    sigma2_theta_new = sample_from_inverse_gamma(a_theta_post, b_theta_post, seed)
    return mu_theta_new, sigma2_theta_new

# training loop

def train_fm_gibbs(
    model_params: dict,
    latent_params: dict,
    fixed_params: dict,
    x_data: np.ndarray,
    y_data: np.ndarray,
    f_init: np.ndarray,
    num_iter: int,
    seed: Optional[int]=None
):
    loss_hist = []

    N = x_data.shape[1]
    K = model_params['v'].shape[1]
    D = x_data.shape[0]

    f = f_init
    q = calc_q_init(x_data, model_params['v'])

    for iter in range(num_iter):
        # noise parameter
        x_b, y_b = calc_xy_b(f, model_params['b'], x_data, y_data)
        latent_params['sigma2_noise'] = sample_posterior_sigma2_noise(x_b, y_b, model_params['b'], fixed_params['a_noise'], fixed_params['b_noise'])

        # hyperparameters
        latent_params['mu_w'], latent_params['sigma2_w'] = sample_posterior_mu_theta_and_sigma2_theta(model_params['w'], latent_params['sigma2_w'], fixed_params['m_w'], fixed_params['l_w'], fixed_params['a_w'], fixed_params['b_w'])
        for k in range(K):
            latent_params['mu_v'][k], latent_params['sigma2_v'][k] = sample_posterior_mu_theta_and_sigma2_theta(model_params['v'][:, k], latent_params['sigma2_v'][k], fixed_params['m_v'], fixed_params['l_v'], fixed_params['a_v'], fixed_params['b_v'])

        # b
        x_b, y_b = calc_xy_b(f, model_params['b'], x_data, y_data)
        b_new    = sample_posterior_theta(x_b, y_b, latent_params['sigma2_noise'], fixed_params['mu_b'], fixed_params['sigma2_b'])
        f        = f + calc_df(x_b, b_new, model_params['b'])
        model_params['b'] = b_new

        # w
        for i in range(N):
            x_w, y_w = calc_xy_w(f, model_params['w'], x_data, y_data, i)
            w_i_new  = sample_posterior_theta(x_w, y_w, latent_params['sigma2_noise'], latent_params['mu_w'], latent_params['sigma2_w'])
            f        = f + calc_df(x_w, w_i_new, model_params['w'][i])
            model_params['w'][i] = w_i_new

        # v
            for k in range(K):
                x_v, y_v = calc_xy_v(f, q, model_params['v'], x_data, y_data, i, k)
                v_ik_new = sample_posterior_theta(x_v, y_v, latent_params['sigma2_noise'], latent_params['mu_v'][k], latent_params['sigma2_v'][k])
                # v_new  = sample_param_lstsq(x_v, y_v)
                f      = f      + calc_df(x_v, v_ik_new, model_params['v'][i, k])
                q[:,k] = q[:,k] + calc_dq(i, x_data, v_ik_new, model_params['v'][i, k])
                model_params['v'][i, k] = v_ik_new

        if (iter+1) % 50 == 0:
            print(f'iter: {iter+1}, loss: {np.sum((y_data - f) ** 2) / D}')
        loss_hist.append(np.sum((y_data - f) ** 2) / D)

    return model_params, loss_hist

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

_, loss_hist = train_fm_gibbs(
    model_params,
    latent_params,
    fixed_params,
    x_data,
    y_data,
    fm.predict(x_data),
    1000,
)

plt.plot(loss_hist)
plt.yscale('log')
plt.show()
