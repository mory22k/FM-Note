1. Dataset:

$$
\begin{aligned}
p(\{ y_\theta^{(d)} \} | \{ x_\theta^{(d)} \}, \Theta, \lambda_n)
&= \prod_{d=1}^D \mathcal{N} \left( y_\theta^{(d)} \middle| \theta x_\theta^{(d)}, \sigma_n^2 \right),
& \theta &= b, w_i, v_{ik}
\end{aligned}
$$

2. Model parameters:

$$
\begin{aligned}
p(b | \mu_b, \lambda_b)
&= \mathcal{N}\left( b \middle| \mu_b, \sigma_b^2 \right),
\\

p(w_i | \mu_w, \lambda_w)
&= \mathcal{N}\left( w_i \middle| \mu_w, \sigma_w^2 \right),
\\

p(v_{ik} | \mu_{v_k}, \lambda_{v_k})
&= \mathcal{N}\left( v_{ik} \middle| \mu_{v_k}, \sigma_{v_k}^2 \right),
\\

p(\sigma_n^2 | \alpha_n, \beta_n)
&= \Gamma^{-1}\left( \sigma_n^2 \middle| \alpha_n, \beta_n \right),
\end{aligned}
$$

3. Latent parameters:

$$
\begin{aligned}
p(\mu_w | \mu_0, \lambda_w, \gamma_0)
&= \mathcal{N}\left( \mu_w \middle| \mu_0, \frac{\sigma_w^2}{\gamma_0} \right),
\\

p(\mu_{v_k} | \mu_0, \lambda_{v_k}, \gamma_0)
&= \mathcal{N}\left( \mu_{v_k} \middle| \mu_0, \frac{\sigma_{v_k}^2}{\gamma_0} \right),
\\

p(\sigma_w^2 | \alpha_0, \beta_0)
&= \Gamma^{-1}\left( \sigma_w^2 \middle| \alpha_0, \beta_0 \right),
\\

p(\sigma_{v_k}^2 | \alpha_0, \beta_0)
&= \Gamma^{-1}\left( \sigma_{v_k}^2 \middle| \alpha_0, \beta_0 \right).
\end{aligned}
$$

Thanks to the conjugacy of the normal and gamma distributions, the posterior distributions of the parameters are also given by normal and gamma distributions:

1. Model parameters:

    $$
    \begin{aligned}
    p(b | \mathcal D, \Theta \setminus \{ b \})
    &= \mathcal{N}\left( b \middle| \mu_b^\star, \sigma_b^{\star 2} \right),
    \\

    p(w_i | \mathcal D, \Theta \setminus \{ w_i \})
    &= \mathcal{N}\left( w_i \middle| \mu_{w_i}^\star, \sigma_{w_i}^{\star 2} \right),
    \\

    p(v_{ik} | \mathcal D, \Theta \setminus \{ v_{ik} \})
    &= \mathcal{N}\left( v_{ik} \middle| \mu_{v_{ik}}^\star, \sigma_{v_{ik}}^{\star 2} \right),
    \\
    p(\sigma_n^2 | \mathcal D)
    &= \Gamma^{-2}\left( \sigma_n^2 \middle| \alpha_n^\star, \beta_n^\star \right),
    \end{aligned}
    $$

    where

    $$
    \begin{aligned}
    \sigma_b^{\star 2} &= \left( \frac{1}{\sigma_n^2} + \frac{1}{\sigma_b^2} \sum_{d=1}^D 1 \right)^{-1},
    &
    \mu_b^\star &= \sigma_b^{\star 2} \left( \frac{1}{\sigma_n^2} \sum_{d=1}^D y_b^{(d)} + \frac{1}{\sigma_b^2} \mu_b \right),
    \\

    \sigma_{w_i}^{\star 2} &= \left( \frac{1}{\sigma_n^2} + \frac{1}{\sigma_w^2} \sum_{d=1}^D x_{w_i}^{(d)2} \right)^{-1},
    &
    \mu_{w_i}^\star &= \sigma_{w_i}^{\star 2} \left( \frac{1}{\sigma_n^2} \sum_{d=1}^D y_{w_i}^{(d)} x_{w_i}^{(d)} + \frac{1}{\sigma_w^2} \mu_w \right),
    \\

    \sigma_{v_{ik}}^{\star 2} &= \left( \frac{1}{\sigma_n^2} + \frac{1}{\sigma_{v_k}^2} \sum_{d=1}^D x_{v_{ik}}^{(d)2} \right)^{-1},
    &
    \mu_{v_{ik}}^\star &= \sigma_{v_{ik}}^{\star 2} \left( \frac{1}{\sigma_n^2} \sum_{d=1}^D y_{v_{ik}}^{(d)} x_{v_{ik}}^{(d)} + \frac{1}{\sigma_{v_k}^2} \mu_{v_k} \right),
    \\

    \alpha_n^\star &= \alpha_n + \frac{D}{2},
    &
    \beta_n^\star &= \beta_n + \frac{1}{2} \sum_{d=1}^D \left( y_b^{(d)} - \theta x_\theta^{(d)} \right)^2.
    \end{aligned}
    $$

2. Latent parameters:

    $$
    \begin{aligned}
    p(\mu_w | \Theta \setminus \{ \mu_w \})
    &= \mathcal{N}\left( \mu_w \middle| \mu_{\mu_w}^\star, \frac{\sigma_0^{2}}{\gamma_{\mu_w}^\star} \right),
    &
    p(\sigma_w^2 | \Theta \setminus \{ \sigma_w^2\})
    &= \Gamma^{-1}\left( \sigma_w^2 \middle| \alpha_{w}^\star, \beta_{w}^\star \right),
    \\

    p(\mu_{v_k} | \Theta \setminus \{ \mu_{v_k} \})
    &= \mathcal{N}\left( \mu_{v_k} \middle| \mu_{\mu_{v_k}}^\star, \frac{\sigma_0^{2}}{\gamma_{\mu_{v_k}}^\star} \right),
    &
    p(\sigma_{v_k}^2 | \Theta \setminus \{ \sigma_{v_k}^2\})
    &= \Gamma^{-1}\left( \sigma_{v_k}^2 \middle| \alpha_{v_{ik}}^\star, \beta_{v_{ik}}^\star \right).
    \end{aligned}
    $$

    where

    $$
    \begin{aligned}
    \mu_{\mu_w}^\star &= \frac{1}{\gamma_{\mu_w}^\star} \left( \sum_{i=1}^N w_i + \gamma_0 \mu_0 \right),
    &
    \gamma_{\mu_w}^\star &= \gamma_0 + N,
    \\

    \mu_{\mu_{v_k}}^\star &= \frac{1}{\gamma_{\mu_{v_k}}^\star} \left( \sum_{i=1}^N v_{ik} + \gamma_0 \mu_0 \right),
    &
    \gamma_{\mu_{v_k}}^\star &= \gamma_0 + N,
    \\

    \alpha_{w_i}^\star &= \alpha_{w} + \frac{N + 1}{2},
    &
    \beta_{w_i}^\star &= \beta_{w} + \frac{1}{2} \left( \sum_{d=1}^D \left( w_i - \mu_w \right)^2 + \gamma_0 \left( \mu_w - \mu_0 \right)^2 \right),
    \\

    \alpha_{v_{ik}}^\star &= \alpha_{v_k} + \frac{N + 1}{2},
    &
    \beta_{v_{ik}}^\star &= \beta_{v_k} + \frac{1}{2} \left( \sum_{d=1}^D \left( v_{ik} - \mu_{v_k} \right)^2 + \gamma_0 \left( \mu_{v_k} - \mu_0 \right)^2 \right).
    \end{aligned}
    $$

If we write $\theta = b, w, v_k$, the above equations can be written in a more compact form:

$$
\begin{aligned}

p(\theta_i | \mathcal D, \Theta \setminus \{ \theta_i \})
&= \mathcal{N}\left( \theta_i \middle| \mu_\theta^\star, \sigma_\theta^{\star 2} \right),
\\

p(\mu_\theta | \Theta \setminus \{ \mu_\theta \})
&= \mathcal{N}\left( \mu_\theta \middle| \mu_{\mu_\theta}^\star, \frac{\sigma_0^{2}}{\gamma_{\mu_\theta}^\star} \right),
\\

p(\sigma_\theta^2 | \Theta \setminus \{ \sigma_\theta^2\})
&= \Gamma^{-1}\left( \sigma_\theta^2 \middle| \alpha_{\theta}^\star, \beta_{\theta}^\star \right).
\end{aligned}
$$

where

$$
\begin{aligned}
\sigma_\theta^{\star 2} &= \left( \frac{1}{\sigma_n^2} + \frac{1}{\sigma_\theta^2} \sum_{d=1}^D x_{\theta_i}^{(d)2} \right)^{-1},
&
\mu_\theta^\star &= \sigma_\theta^{\star 2} \left( \frac{1}{\sigma_n^2} \sum_{d=1}^D y_{\theta_i}^{(d)} x_{\theta_i}^{(d)} + \frac{1}{\sigma_\theta^2} \mu_\theta \right),
\\

\mu_{\mu_\theta}^\star &= \frac{1}{\gamma_{\mu_\theta}^\star} \left( \sum_{i=1}^N \theta_i + \gamma_0 \mu_0 \right),
&
\gamma_{\mu_\theta}^\star &= \gamma_0 + N,
\\

\alpha_{\theta}^\star &= \alpha_{\theta} + \frac{N + 1}{2},
&
\beta_{\theta}^\star &= \beta_{\theta} + \frac{1}{2} \left( \sum_{d=1}^D \left( \theta_i - \mu_\theta \right)^2 + \gamma_0 \left( \mu_\theta - \mu_0 \right)^2 \right).
\\

\alpha_n^\star &= \alpha_n + \frac{D}{2},
&
\beta_n^\star &= \beta_n + \frac{1}{2} \sum_{d=1}^D \left( y_b^{(d)} - \theta x_\theta^{(d)} \right)^2.
\end{aligned}
$$
