using LinearAlgebra
using Random
using Distributions

function sample_gaussian_efficient(
    Phi,
    alpha,
    Delta,
)
    D, N = size(Phi)

    # Sample u from N(0, Delta)
    if isapprox(Delta, Diagonal(diag(Delta)))
        u = randn(N) .* sqrt.(diag(Delta))
    else
        u = rand(MvNormal(zeros(N), Delta))
    end

    # Sample v from N(Phi * u, I)
    v = Phi * u + randn(D)

    # Solve for w
    w = Phi * Delta * Phi' + I
    w = w \ (alpha - v)

    return u + Delta * Phi' * w
end

function sample_w_posterior(
    x_data,
    y_data,
    L,
    sigma2_n,
)
    A = transpose(x_data) * x_data + Diagonal(1 ./ L)
    A_inv = inv(A)
    m = A_inv * transpose(x_data) * y_data
    V = sigma2_n * A_inv

    try
        return rand(MvNormal(m, V))
    catch _
        return rand(MvNormal(m, V + I*1e-10))
    end
end

function sample_w_posterior_efficient(
    x_data,
    y_data,
    L,
    sigma2_n,
)
    Phi = x_data ./ sqrt(sigma2_n)
    alpha = y_data ./ sqrt(sigma2_n)
    Delta = sigma2_n * Diagonal(L)
    return sample_gaussian_efficient(Phi, alpha, Delta)
end

function sample_sigma2_n_posterior(
    x_data,
    y_data,
    w,
    L,
)
    N = length(w)
    D = length(y_data)

    error = y_data - x_data * w

    shape = (N + D) / 2
    scale = dot(error, error) / 2 + sum(w.^2 ./ L) / 2

    result = rand(InverseGamma(shape, scale))
    return result
end

function sample_lamb2_posterior(
    w,
    tau2,
    nu,
    sigma2_n,
)
    result = similar(w)
    for i in eachindex(w)
        scale = 1. / nu[i] + w[i]^2 / (sigma2_n * tau2 * 2)
        result[i] = rand(InverseGamma(1., scale))
    end
    return result
end

function sample_tau2_posterior(
    w,
    lamb2,
    xi,
    sigma2_n,
)
    N = length(w)
    shape = (N + 1) / 2
    scale = 1. / xi + sum(w.^2 ./ lamb2) / (sigma2_n * 2)

    result = rand(InverseGamma(shape, scale))
    return result
end

function sample_nu_posterior(
    lamb2,
)
    result = similar(lamb2)
    for i in eachindex(result)
        scale = 1. + 1. / lamb2[i]
        result[i] = rand(InverseGamma(1., scale))
    end
    return result
end

function sample_xi_posterior(
    tau2,
)
    scale = 1. + 1. / tau2

    result = rand(InverseGamma(1., scale))
    return result
end

function sample_params_posterior_efficient(
    x_data,
    y_data,
    w,
    sigma2_n,
    lamb2,
    tau2,
    nu,
    xi,
)
    D, N = size(x_data)
    L = lamb2 .* tau2

    try
        w = sample_w_posterior_efficient(x_data, y_data, L, sigma2_n)
    catch _
        try
            w = sample_w_posterior(x_data, y_data, L, sigma2_n)
        catch _
            # println("error while sampling w")
            w = w
        end
    end

    sigma2_n = sample_sigma2_n_posterior(x_data, y_data, w, L)
    lamb2 = sample_lamb2_posterior(w, tau2, nu, sigma2_n)
    tau2 = sample_tau2_posterior(w, lamb2, xi, sigma2_n)
    nu = sample_nu_posterior(lamb2)
    xi = sample_xi_posterior(tau2)
    return w, sigma2_n, lamb2, tau2, nu, xi
end

function sample_horseshoe_posterior(
    x_data,
    y_data,
    num_iter,
    show_progress=false,
)
    N = size(x_data, 2)

    w = zeros(N)
    sigma2_n = 1.0
    lamb2 = ones(N)
    tau2 = 1.0
    nu = ones(N)
    xi = 1.0

    for t in 1:num_iter
        w, sigma2_n, lamb2, tau2, nu, xi = sample_params_posterior_efficient(
            x_data, y_data, w, sigma2_n, lamb2, tau2, nu, xi
        )
        if show_progress && (t % 100) == 0
            y_pred = x_data * w
            MSE = sum((y_data - y_pred).^2) / length(y_data)
            println("iter $(t) | MSE: $(MSE)")
        end
    end

    return w
end

function sample_from_inverse_gamma(
    shape,
    scale,
)
    return rand(InverseGamma(shape, scale))
end

function sample_horseshoe_posterior_old(X, Y, num_iter, show_progress)
    d, n = size(X)
    XtX = X' * X
    XtY = X' * Y

    w = zeros(n)
    sigma2 = 1.0
    lamb2 = ones(n)
    tau2 = 1.0
    nu = ones(n)
    xi = 1.0

    error_history = zeros(num_iter)
    for i in 1:num_iter
        if d <= n
            try
                w = normal_efficient_sampler(X, tau2 .* lamb2, Y ./ sqrt(sigma2)) .* sqrt(sigma2)
            catch _
                S_inv = Diagonal(1.0 ./ (tau2 .* lamb2))
                A = XtX + S_inv
                A_inv = inv(A)
                try
                    w = rand(MvNormal(A_inv * XtY, sigma2 * A_inv + 1e-10 * I))
                catch _
                    # println("error while sampling w")
                    w = rand(MvNormal(A_inv * XtY, 1e-10 * I))
                end
            end
        else
            S_inv = Diagonal(1.0 ./ (tau2 .* lamb2))
            A = XtX + S_inv
            A_inv = inv(A)
            try
                w = rand(MvNormal(A_inv * XtY, sigma2 * A_inv))
            catch _
                try
                    w = normal_efficient_sampler(X, tau2 .* lamb2, Y ./ sqrt(sigma2)) .* sqrt(sigma2)
                catch _
                    # println("error while sampling w")
                    w = rand(MvNormal(A_inv * XtY, 1e-10 * I))
                end
            end
        end

        # Sample sigma^2
        e = Y - X * w
        shape = (d + n) / 2
        scale = dot(e, e) / 2 + sum(w .^ 2 ./ lamb2) / (tau2 * 2)
        sigma2 = sample_from_inverse_gamma(shape, scale)

        # Sample beta^2
        for k in 1:n
            scale = 1. / nu[k] + w[k]^2 / (2 * tau2 * sigma2)
            lamb2[k] = sample_from_inverse_gamma(1., scale)
        end

        # Sample tau^2
        shape = (n + 1) / 2
        scale = 1. / xi + sum(w .^ 2 ./ lamb2) / (2 * sigma2)
        tau2 = sample_from_inverse_gamma(shape, scale)

        # Sample nu
        for k in 1:n
            scale = 1. + 1. / lamb2[k]
            nu[k] = sample_from_inverse_gamma(1., scale)
        end

        # Sample xi
        scale = 1. + 1. / tau2
        xi = sample_from_inverse_gamma(1., scale)

        if show_progress
            println("Iteration $i: $(mean(e .^ 2))")
        end
    end

    return w
end
