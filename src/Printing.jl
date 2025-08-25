# Terminal output

function printStart()
    print_dashed_line()
    @printf(
        stdout,
        "|%7s %9s|%19s %19s|%7s %7s %7s %7s %7s|%7s|%10s %7s %6s|%8s|%7s %7s %7s %7s|\n",
        "iter",
        "secs",
        "primal",
        "dual",
        "gap",
        "pinf",
        "dinf",
        "compl",
        "compl*",
        "mu",
        "evals",
        "avg",
        "max",
        "nnz(y)",
        "p_rat",
        "d_rat",
        "sum",
        "ratio"
    )
    return print_dashed_line()
end

function printUpdate(
    iter::Int64,
    secs::Float64,
    primal_obj::T,
    dual_obj::T,
    gap::T,
    pinf::T,
    dinf::T,
    compl::T,
    compl_not_psd_abs::T,
    point::Point{T},
    done::Bool,
    total_sum_evals::Int64,
    last_iter_evals::Vector{Int64},
    last_max_evals::Vector{Int64},
    norm_p_res::T,
    norm_d_res::T,
    use_double_sweep::Bool,
) where {T<:AbstractFloat}
    @assert iter >= 0
    @assert secs >= 0
    @assert gap >= 0
    @assert pinf >= 0
    @assert dinf >= 0
    @assert compl >= 0
    @assert compl_not_psd_abs >= 0
    @assert point.mu > 0
    @assert total_sum_evals >= 0
    @assert minimum(last_iter_evals) >= 0
    @assert length(last_iter_evals) == length(last_max_evals)
    @assert minimum(last_max_evals) >= 0

    n::Int64 = sum(size(V, 2) for V in point.Vs)
    avg_evals::Float64 = compute_avg_evals(
        use_double_sweep ? 2 * n : n, iter, last_iter_evals
    )

    if (iter <= length(last_iter_evals) || mod(iter, length(last_iter_evals)) == 0 || done)
        @printf(
            stdout,
            "|%7d %9.1f|%19.12e %19.12e|%7.1e %7.1e %7.1e %7.1e %7.1e|%7.1e|%10d %7.1f %6d|%8d|%7.1e %7.1e %7.1e %7.1e|\n",
            iter,
            secs,
            primal_obj,
            dual_obj,
            gap,
            pinf,
            dinf,
            compl,
            compl_not_psd_abs,
            point.mu,
            total_sum_evals,
            avg_evals,
            maximum(last_max_evals),
            count(!iszero, point.y),
            norm_p_res,
            norm_d_res,
            norm_p_res + norm_d_res,
            norm_p_res / norm_d_res
        )
    end

end

function print_dashed_line()
    return print(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
    )
end

function compute_avg_evals(n::Int64, iter::Int64, last_iter_evals::Vector{Int64})
    @assert n >= 1
    @assert iter >= 0
    @assert length(last_iter_evals) >= 1
    @assert minimum(last_iter_evals) >= 0
    avg_evals::Float64 = 0.0 # for iter == 0
    if 1 <= iter && iter <= length(last_iter_evals) - 1
        avg_evals = 1.0 * sum(last_iter_evals) / n / iter
    else
        avg_evals = 1.0 * sum(last_iter_evals) / n / length(last_iter_evals)
    end
    return avg_evals
end
