using LinearAlgebra

function calc_q_init(x, v)::Array{Float64,2}
    # x: (D, N)
    # v: (N, K)
    return x[:, :] * v[:, :] # (D, K)
end

function calc_dq_d(x, v_ik_new, v, i, k, d)
    # v_ik_new: float
    # v: (N, K)
    # x: (D, N)
    return (v_ik_new - v[i, k]) * x[d, i]
end

function calc_df_d(h, p_new, p_old, d)
    # h: (D)
    return (p_new - p_old) * h[d]
end

function calc_h_b_d(x)
    return 1.0
end

function calc_h_w_d(x, i, d)
    # x: (D, N)
    return x[d, i]
end

function calc_h_v_fast_d(x, v, q, i, k, d)
    # x: (D, N)
    # v: (N, K)
    # q: (D, K)
    return x[d, i] * (q[d, k] - x[d, i] * v[i, k])
end

function calc_g_d(f, h_d, p, d)
    # f: (D)
    # h: (D)
    return f[d] - h_d[d] * p
end

function sample_param_lstsq(y, h, g, lamb=1e-8)
    # h: (D)
    # g: (D)
    x_theta = h
    y_theta = y - g
    return sum(x_theta .* y_theta) / (sum(x_theta .^ 2) + lamb)
end

function train_fm_als_d(
    init_params::Dict{Any, Any},
    x_data::Array{Int64, 2},
    y_data::Array{Float64, 1},
    f_init::Array{Float64, 1},
    num_iter::Int
)::Dict{String, Any}
    # get indices
    N = size(x_data, 2)
    K = size(init_params["v"], 2)
    D = size(x_data, 1)

    # get initial parameter
    params = copy(init_params)

    # prepare placeholders
    h_b = zeros(D)
    g_b = zeros(D)
    h_w = zeros(D)
    g_w = zeros(D)
    h_v = zeros(D)
    g_v = zeros(D)

    # precalculate
    f = copy(f_init)
    q = calc_q_init(x_data, params["v"])

    # main loop
    for iter in 1:num_iter
        # sample b
        for d in 1:D
            h_b[d] = calc_h_b_d(x_data)
            g_b[d] = calc_g_d(f, h_b, params["b"], d)
        end
        b_new = sample_param_lstsq(y_data, h_b, g_b)
        for d in 1:D
            f[d] += calc_df_d(h_b, b_new, params["b"], d)
        end
        params["b"] = b_new

        # sample w
        for i in 1:N
            for d in 1:D
                h_w[d] = calc_h_w_d(x_data, i, d)
                g_w[d] = calc_g_d(f, h_w, params["w"][i], d)
            end
            w_i_new = sample_param_lstsq(y_data, h_w, g_w)
            for d in 1:D
                f[d] += calc_df_d(h_w, w_i_new, params["w"][i], d)
            end
            params["w"][i] = w_i_new
        end

        # sample v
        for i in 1:N
            for k in 1:K
                for d in 1:D
                    h_v[d] = calc_h_v_fast_d(x_data, params["v"], q, i, k, d)
                    g_v[d] = calc_g_d(f, h_v, params["v"][i, k], d)
                end
                v_ik_new = sample_param_lstsq(y_data, h_v, g_v)
                for d in 1:D
                    f[d] += calc_df_d(h_v, v_ik_new, params["v"][i, k], d)
                    q[d, k] += calc_dq_d(x_data, v_ik_new, params["v"], i, k, d)
                end
                params["v"][i, k] = v_ik_new
            end
        end

        if iter % 10 == 0
            println("iter: $iter, loss: $(sum((y_data - f) .^ 2) / D)")
        end
    end

    return params
end
