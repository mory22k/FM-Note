function calc_q_init(x::Array{Int64,2}, v::Array{Float64,2})::Array{Float64,2}
    # x: (D, N)
    # v: (N, K)
    return x[:, :] * v[:, :] # (D, K)
end

function calc_dq(x::Array{Int64,2}, v_ik_new::Float64, v::Array{Float64,2}, i::Int, k::Int)::Array{Float64,1}
    # v_ik_new: float
    # v: (N, K)
    # x: (D, N)
    return (v_ik_new - v[i, k]) * x[:, i] # (D)
end

function calc_df(h::Array{Float64,1}, p_new::Float64, p_old::Float64)::Array{Float64,1}
    # h: (D)
    return (p_new - p_old) * h # (D)
end

function calc_h_b(x::Array{Int64,2})::Array{Float64,1}
    return ones(size(x, 1))
end

function calc_h_w(x::Array{Int64,2}, i::Int)::Array{Float64,1}
    # x: (D, N)
    return x[:, i]
end

function calc_h_v_fast(x::Array{Int64,2}, v::Array{Float64,2}, q::Array{Float64,2}, i::Int, k::Int)::Array{Float64,1}
    # x: (D, N)
    # v: (N, K)
    # q: (D, K)
    return x[:, i] .* (q[:, k] - x[:, i] .* v[i, k]) # (D)
end

function calc_g(f::Array{Float64,1}, h::Array{Float64,1}, p::Float64)::Array{Float64,1}
    g = f - h .* p
    return g
end

function sample_param_lstsq(y::Array{Float64,1}, h::Array{Float64,1}, g::Array{Float64,1}, lamb::Float64=1e-8)::Float64
    # h: (D)
    # g: (D)
    x_theta = h
    y_theta = y - g
    return sum(x_theta .* y_theta) / (sum(x_theta .^ 2) + lamb)
end

function train_fm_als(
    init_params::Dict,
    x_data::Array{Int64,2},
    y_data::Array{Float64,1},
    f_init::Array{Float64,1},
    num_iter::Int
)::Dict
    # get indices
    N = size(x_data, 2)
    K = size(init_params["v"], 2)

    # get initial parameter
    params = init_params

    # precalculate
    f = f_init
    q = calc_q_init(x_data, params["v"])

    # main loop
    for iter in 1:num_iter
        # sample b
        h_b   = calc_h_b(x_data)
        g_b   = calc_g(f, h_b, params["b"])
        b_new = sample_param_lstsq(y_data, h_b, g_b)
        f     = f + calc_df(h_b, b_new, params["b"])
        params["b"] = b_new

        # sample w
        for i in 1:N
            h_w   = calc_h_w(x_data, i)
            g_w   = calc_g(f, h_w, params["w"][i])
            w_i_new = sample_param_lstsq(y_data, h_w, g_w)
            f     = f + calc_df(h_w, w_i_new, params["w"][i])
            params["w"][i] = w_i_new
        end

        # sample v
        for i in 1:N
            for k in 1:K
                h_v      = calc_h_v_fast(x_data, params["v"], q, i, k)
                g_v      = calc_g(f, h_v, params["v"][i, k])
                v_ik_new = sample_param_lstsq(y_data, h_v, g_v)
                f        = f      + calc_df(h_v, v_ik_new, params["v"][i, k])
                q[:,k]   = q[:,k] + calc_dq(x_data, v_ik_new, params["v"], i, k)
                params["v"][i, k] = v_ik_new
            end
        end

        if iter % 10 == 0
            println("iter: $iter, loss: $(sum((y_data - f) .^ 2) / size(y_data, 1))")
        end
    end

    return params
end
