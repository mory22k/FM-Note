function calc_q_init(
    x::Array{Float64},
    v::Array{Float64}
) :: Array{Float64}
    # x: (D, N)
    # v: (N, K)
    return x[:, :] * v[:, :] # (D, K)
end

function calc_dq(
    i::Int,
    x::Array{Float64},
    v_ik_new::Float64,
    v_ik_old::Float64
) :: Array{Float64, 1}
    # v_ik_new: float
    # v: (N, K)
    # x: (D, N)
    return (v_ik_new - v_ik_old) .* x[:, i] # (D)
end

function calc_df(
    x_theta::Array{Float64},
    param_new::Float64,
    param_old::Float64
)
    return (param_new - param_old) .* x_theta
end

function calc_xy_b(
    f::Array{Float64, 1},
    b::Float64,
    x_data::Array{Float64},
    y_data::Array{Float64, 1}
)
    # x_data: (D, N)
    # y_data: (D)
    x_b = ones(size(x_data, 1))
    y_b = y_data - (f - b .* x_b)
    return x_b, y_b
end

function calc_xy_w(
    f::Array{Float64, 1},
    w::Array{Float64, 1},
    x_data::Array{Float64},
    y_data::Array{Float64, 1},
    i::Int
)
    # x_data: (D, N)
    # y_data: (D)
    x_w = x_data[:, i]
    y_w = y_data - (f - x_w .* w[i])
    return x_w, y_w
end

function calc_xy_v(
    f::Array{Float64, 1},
    q::Array{Float64},
    v::Array{Float64},
    x_data::Array{Float64},
    y_data::Array{Float64, 1},
    i::Int,
    k::Int
)
    # x_data: (D, N)
    # y_data: (D)
    x_v = x_data[:, i] .* (q[:, k] - x_data[:, i] .* v[i, k])
    y_v = y_data - (f - x_v .* v[i, k])
    return x_v, y_v
end

function sample_param_lstsq(
    x_theta::Array{Float64},
    y_theta::Array{Float64},
    lamb::Float64=1e-8
) :: Float64
    return sum(x_theta .* y_theta) / (sum(x_theta .^ 2) + lamb)
end

function train_fm_als(
    init_params::Dict{Any, Any},
    x_data::Array{Float64},
    y_data::Array{Float64, 1},
    f_init::Array{Float64, 1},
    num_iter::Int
) :: Dict{Any, Any}
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
        x_b, y_b = calc_xy_b(f, params["b"], x_data, y_data)
        b_new    = sample_param_lstsq(x_b, y_b)
        f        = f + calc_df(x_b, b_new, params["b"])
        params["b"] = b_new

        # sample w
        for i in 1:N
            x_w, y_w = calc_xy_w(f, params["w"], x_data, y_data, i)
            w_i_new  = sample_param_lstsq(x_w, y_w)
            f        = f + calc_df(x_w, w_i_new, params["w"][i])
            params["w"][i] = w_i_new

            # sample v
            for k in 1:K
                x_v, y_v = calc_xy_v(f, q, params["v"], x_data, y_data, i, k)
                v_ik_new = sample_param_lstsq(x_v, y_v)
                f        = f      + calc_df(x_v, v_ik_new, params["v"][i, k])
                q[:,k]   = q[:,k] + calc_dq(i, x_data, v_ik_new, params["v"][i, k])
                params["v"][i, k] = v_ik_new
            end
        end

        if iter % 10 == 0
            println("iter: $iter, loss: $(sum((y_data - f) .^ 2) / N)")
        end
    end

    return params
end
