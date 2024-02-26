using Random
using Distributions
using LinearAlgebra

function sample_gaussian_efficient(
    Phi,
    alpha,
    Delta,
)
    N_data, N_vars = size(Phi)
    if ndims(Delta) == 1
        u = randn(N_vars) .* sqrt.(Delta)
        Delta = Diagonal(Delta)
    else
        u = rand(MvNormal(zeros(N_vars), Delta))
    end

    delta = randn(N_data)
    v = Phi * u + delta
    w = (Phi * Delta * Phi' + I) \ (alpha - v)
    w_sample = u + Delta * Phi' * w
    return w_sample
end

function sample_w_posterior(x_data, y_data, N_data, N_vars, XtY, XtX, tau2, lamb2, sigma2)
    if N_data <= N_vars
        try
            w = sample_gaussian_efficient(x_data ./ sqrt(sigma2), y_data ./ sqrt(sigma2), tau2 .* lamb2 .* sigma2)
        catch _
            A = XtX + Diagonal(1.0 ./ (tau2 .* lamb2))
            A_inv = inv(A)
            try
                w = rand(MvNormal(A_inv * XtY, sigma2 * A_inv + 1e-10 * I))
            catch _
                println("error while sampling w")
                w = rand(MvNormal(A_inv * XtY, 1e-10 * I))
            end
        end
    else
        A = XtX + Diagonal(1.0 ./ (tau2 .* lamb2))
        A_inv = inv(A)
        try
            w = rand(MvNormal(A_inv * XtY, sigma2 * A_inv))
        catch _
            try
                w = sample_gaussian_efficient(x_data ./ sqrt(sigma2), y_data ./ sqrt(sigma2), tau2 .* lamb2 .* sigma2)
            catch _
                println("error while sampling w")
                w = rand(MvNormal(A_inv * XtY, 1e-10 * I))
            end
        end
    end
end

function sample_sigma2_n_posterior(x_data, y_data, w, N_data, N_vars, lamb2, tau2)
    error = y_data - x_data * w
    shape = (N_data + N_vars) / 2
    scale = dot(error, error) / 2 + sum(w .^ 2 ./ lamb2) / (tau2 * 2)
    return rand(InverseGamma(shape, scale))
end

function sample_lamb2_posterior(w, N_vars, sigma2_n, lamb2, tau2, nu)
    shape = 1.
    for k in 1:N_vars
        scale = 1. / nu[k] + w[k]^2 / (2 * tau2 * sigma2_n)
        lamb2[k] = rand(InverseGamma(shape, scale))
    end
    return lamb2
end

function sample_tau2_posterior(w, N_vars, sigma2_n, lamb2, xi)
    shape = (N_vars + 1) / 2
    scale = 1. / xi + sum(w .^ 2 ./ lamb2) / (2 * sigma2_n)
    return rand(InverseGamma(shape, scale))
end

function sample_nu_posterior(N_vars, lamb2, nu)
    shape = 1.
    for k in 1:N_vars
        scale = 1. + 1. / lamb2[k]
        nu[k] = rand(InverseGamma(shape, scale))
    end
    return nu
end

function sample_xi_posterior(tau2)
    shape = 1.
    scale = 1. + 1. / tau2
    return rand(InverseGamma(shape, scale))
end

function sample_horseshoe_posterior(X, Y, num_iter, show_progress, seed=nothing, return_error_history=false)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    N_data, N_vars = size(X)

    w        = zeros(N_vars)
    sigma2_n = 1.
    lamb2    = ones(size(X, 2))
    tau2     = 1.
    nu       = ones(size(X, 2))
    xi       = 1.

    XtX = X' * X
    XtY = X' * Y

    for i in 1:num_iter
        w        = sample_w_posterior(X, Y, N_data, N_vars, XtY, XtX, tau2, lamb2, sigma2_n)
        sigma2_n = sample_sigma2_n_posterior(X, Y, w, N_data, N_vars, lamb2, tau2)
        lamb2    = sample_lamb2_posterior(w, N_vars, sigma2_n, lamb2, tau2, nu)
        tau2     = sample_tau2_posterior(w, N_vars, sigma2_n, lamb2, xi)
        nu       = sample_nu_posterior(N_vars, lamb2, nu)
        xi       = sample_xi_posterior(tau2)

        if show_progress && i % 100 == 0
            error = Y - X * w
            println("Iteration $i: $(mean(error .^ 2))")
        end
    end

    return w
end
