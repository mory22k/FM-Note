using Random
using Distributions

# update rules

function calc_q_init(
    x::Array{Float64,2},
    v::Array{Float64,2}
)::Array{Float64,2}
    # x: (D, N)
    # v: (N, K)
    return x[:, :] * v[:, :] # (D, K)
end

function calc_dq(
    i::Int,
    x::Array{Float64,2},
    v_ik_new::Float64,
    v_ik_old::Float64
)::Array{Float64,1}
    # v_ik_new: float
    # v: (N, K)
    # x: (D, N)
    return (v_ik_new - v_ik_old) .* x[:, i] # (D)
end

function calc_df(
    x_theta::Array{Float64,1},
    param_new::Float64,
    param_old::Float64
)
    return (param_new - param_old) * x_theta
end

function calc_xy_b(
    f::Array{Float64,1},
    b::Float64,
    x_data::Array{Float64,2},
    y_data::Array{Float64,1}
)
    # x_data: (D, N)
    # y_data: (D)
    x_b = ones(size(x_data, 1))
    y_b = y_data - (f - b * x_b)
    return x_b, y_b
end

function calc_xy_w(
    f::Array{Float64,1},
    w::Array{Float64,1},
    x_data::Array{Float64,2},
    y_data::Array{Float64,1},
    i::Int
)
    # x_data: (D, N)
    # y_data: (D)
    x_w = x_data[:, i]
    y_w = y_data - (f - x_w .* w[i])
    return x_w, y_w
end

function calc_xy_v(
    f::Array{Float64,1},
    q::Array{Float64,2},
    v::Array{Float64,2},
    x_data::Array{Float64,2},
    y_data::Array{Float64,1},
    i::Int,
    k::Int
)
    # x_data: (D, N)
    # y_data: (D)
    x_v = x_data[:, i] .* (q[:, k] - x_data[:, i] .* v[i, k])
    y_v = y_data - (f - x_v .* v[i, k])
    return x_v, y_v
end

# samplers
function sample_from_inverse_gamma(
    a::Float64,
    b::Float64,
    seed::Union{Int, Nothing}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    return 1 / rand(Gamma(a, 1 / b))
end

function sample_posterior_sigma2_noise(
    x_theta::Array{Float64,1},
    y_theta::Array{Float64,1},
    theta::Float64,
    a_noise::Float64,
    b_noise::Float64,
    seed::Union{Int, Nothing}=nothing
)
    # x_theta: (D)
    # y_theta: (D)
    D = length(x_theta)
    a_noise_post = a_noise + D / 2
    b_noise_post = b_noise + 0.5 * sum((y_theta - x_theta .* theta).^2)
    return sample_from_inverse_gamma(a_noise_post, b_noise_post, seed)
end

function sample_posterior_theta(
    x_theta::Array{Float64,1},
    y_theta::Array{Float64,1},
    sigma2_noise::Float64,
    mu_theta::Float64,
    sigma2_theta::Float64,
    seed::Union{Int, Nothing}=nothing
)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    sigma2_theta_post = 1 / (sum(x_theta.^2) / sigma2_noise + 1 / sigma2_theta)
    mu_theta_post = sigma2_theta_post * (sum(x_theta .* y_theta) / sigma2_noise + mu_theta / sigma2_theta)
    return rand(Normal(mu_theta_post, sqrt(sigma2_theta_post)))
end

function sample_posterior_mu_theta_and_sigma2_theta(
    theta::Array{Float64,1},
    sigma2_theta::Float64,
    m_theta::Float64,
    l_theta::Float64,
    a_theta::Float64,
    b_theta::Float64,
    seed::Union{Int, Nothing}=nothing
)
    # theta: (N)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    N = length(theta)

    l_theta_post = l_theta + N
    mu_theta_post = (sum(theta) + l_theta * m_theta) / l_theta_post
    mu_theta_new  = rand(Normal(mu_theta_post, sqrt(sigma2_theta / l_theta_post)))

    mu_theta = mu_theta_new

    a_theta_post = a_theta + (N + 1) / 2
    b_theta_post = b_theta + 0.5 * (sum((theta .- mu_theta).^2) + l_theta * (mu_theta - m_theta)^2)

    sigma2_theta_new = sample_from_inverse_gamma(a_theta_post, b_theta_post, seed)
    return mu_theta_new, sigma2_theta_new
end

# training loop

function train_fm_gibbs(
    model_params::Dict{Any, Any},
    latent_params::Dict{Any, Any},
    fixed_params::Dict{Any, Any},
    x_data::Array{Float64,2},
    y_data::Array{Float64,1},
    f_init::Array{Float64,1},
    num_iter::Int,
    seed::Union{Int, Nothing}=nothing
)
    loss_hist = []

    N = size(x_data, 2)
    K = size(model_params["v"], 2)
    D = size(x_data, 1)

    f = f_init
    q = calc_q_init(x_data, model_params["v"])

    for iter in 1:num_iter
        # noise parameter
        x_b, y_b = calc_xy_b(f, model_params["b"], x_data, y_data)
        latent_params["sigma2_noise"] = sample_posterior_sigma2_noise(x_b, y_b, model_params["b"], fixed_params["a_noise"], fixed_params["b_noise"])

        # hyperparameters
        latent_params["mu_w"], latent_params["sigma2_w"] = sample_posterior_mu_theta_and_sigma2_theta(model_params["w"], latent_params["sigma2_w"], fixed_params["m_w"], fixed_params["l_w"], fixed_params["a_w"], fixed_params["b_w"])
        for k in 1:K
            latent_params["mu_v"][k], latent_params["sigma2_v"][k] = sample_posterior_mu_theta_and_sigma2_theta(model_params["v"][:, k], latent_params["sigma2_v"][k], fixed_params["m_v"], fixed_params["l_v"], fixed_params["a_v"], fixed_params["b_v"])
        end

        # b
        x_b, y_b = calc_xy_b(f, model_params["b"], x_data, y_data)
        b_new    = sample_posterior_theta(x_b, y_b, latent_params["sigma2_noise"], fixed_params["mu_b"], fixed_params["sigma2_b"])
        f        = f + calc_df(x_b, b_new, model_params["b"])
        model_params["b"] = b_new

        # w
        for i in 1:N
            x_w, y_w = calc_xy_w(f, model_params["w"], x_data, y_data, i)
            w_i_new  = sample_posterior_theta(x_w, y_w, latent_params["sigma2_noise"], latent_params["mu_w"], latent_params["sigma2_w"])
            f        = f + calc_df(x_w, w_i_new, model_params["w"][i])
            model_params["w"][i] = w_i_new
        end

        # v
        for i in 1:N
            for k in 1:K
                x_v, y_v = calc_xy_v(f, q, model_params["v"], x_data, y_data, i, k)
                v_ik_new = sample_posterior_theta(x_v, y_v, latent_params["sigma2_noise"], latent_params["mu_v"][k], latent_params["sigma2_v"][k])
                f      = f      + calc_df(x_v, v_ik_new, model_params["v"][i, k])
                q[:,k] = q[:,k] + calc_dq(i, x_data, v_ik_new, model_params["v"][i, k])
                model_params["v"][i, k] = v_ik_new
            end
        end

        if (iter+1) % 100 == 0
            println("iter: $(iter+1), loss: $(sum((y_data - f) .^ 2) / D)")
        end
        push!(loss_hist, sum((y_data - f) .^ 2) / D)
    end

    return model_params, loss_hist
end
