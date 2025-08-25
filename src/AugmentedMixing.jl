"""
AugmentedMixing.jl - Implementation of the Augmented Mixing Method for large-scale SDPs

This package implements a Burer-Monteiro factorization-based algorithm
for arbitrary semidefinite programs in which the factorization matrices
are updated in a column-wise fashion. Users can call
[`augmented_mixing`](@ref), which solves an SDP specified by a struct [`SdpData`](@ref) and returns
an approximate primal-dual solution together with a [`WarmStart`](@ref)
object to work with extended precision.

# Public API
- [`augmented_mixing`](@ref) - main routine called by user
- [`SdpData`](@ref) - struct to specify the SDP to be solved
- [`WarmStart`](@ref) - struct returned by [`augmented_mixing`](@ref); can be used for doing a warm start

We refer to the preprint https://arxiv.org/abs/2507.20386 for implementation details.
"""
module AugmentedMixing

using LinearAlgebra, Optim, Printf, SparseArrays

export augmented_mixing, WarmStart

"""
    struct WarmStart{T<:AbstractFloat}
        mu::T
        Vs::Vector{Matrix{T}}
        y::Vector{T}

Struct to save a state snapshot for warm-starting [`augmented_mixing`](@ref).
- `mu`: current penalty parameter  
- `Vs`: block-wise Burer-Monteiro factorization matrices (for each SDP block `b`, `X_b = V_b' * V_b`)  
- `y`: dual multipliers for affine constraints (nonnegative for inequalities)

Two constructors are provided:
- `WarmStart(mu, Vs, y)` - not recommended to be used
- `WarmStart(ws::WarmStart{T}, T_new::Type)` - convert a struct WarmStart to another floating-point type (e.g. `Float64` -> `BigFloat`)

A struct WarmStart is returned by [`augmented_mixing`](@ref) and can be fed back via the
`warm_start` keyword to continue a previous run or to refine a solution after promoting to extended precision.
"""
struct WarmStart{T<:AbstractFloat}
    mu::T # penalty parameter
    Vs::Vector{Matrix{T}} # Burer-Monteiro factorization matrices
    y::Vector{T} # dual multipliers

    function WarmStart(mu::T, Vs::Vector{Matrix{T}}, y::Vector{T}) where {T<:AbstractFloat}
        @assert mu > zero(T)
        return new{T}(mu, Vs, y)
    end

    function WarmStart(
        warm_start::WarmStart{T}, T_new::Type{<:AbstractFloat}
    ) where {T<:AbstractFloat}
        return WarmStart(
            map(T_new, warm_start.mu), map.(T_new, warm_start.Vs), map.(T_new, warm_start.y)
        )
    end
end

# stores current iterates and evaluations of linear operators
mutable struct Point{T}
    Xs::Vector{Matrix{T}} # X[b] = V[b]^T * V[b]
    y::Vector{T} # dual variables of affine constraints; length m; nonnegative for inequality constraints
    CX::T # current objective value; CX = <Cs[1], Xs[1]> + ... + <Cs[q], Xs[q]>
    AX::Vector{T} # used to store A(X) = A_1(X_1) + ... + A_q(X_q) where q is the number of blocks
    Vs::Vector{Matrix{T}} # our current iterates Vs[1], Vs[2], ..., Vs[q]
    mu::T # penalty parameter > 0

    function Point{T}(
        ns::Vector{Int64},
        ks::Vector{Int64},
        m::Int64,
        mu_start::T;
        warm_start::Union{WarmStart,Nothing}=nothing,
    ) where {T<:AbstractFloat}
        if isnothing(warm_start)
            Vs::Vector{Matrix{T}} = []
            Xs::Vector{Matrix{T}} = []
            for b in eachindex(ns, ks)
                V::Matrix{T} = randn(T, min(ks[b], ns[b]), ns[b])
                foreach(v -> v .= v ./ norm(v), eachcol(V))
                push!(Vs, V)
                X::Matrix{T} = V' * V
                @assert issymmetric(X)
                push!(Xs, X)
            end
            return new{T}(
                Xs, # X
                zeros(T, m), # y
                zero(T), # CX
                zeros(T, m), # AX
                Vs, # V
                mu_start, # mu
            )

        else # warm start
            @assert warm_start.mu > zero(T)
            @assert length(warm_start.Vs) == length(ns) == length(ks)
            @assert all(
                size(warm_start.Vs[b]) == (ks[b], ns[b]) for
                b in eachindex(warm_start.Vs, ks, ns)
            )
            @assert length(warm_start.y) == m

            return new{T}(
                [V' * V for V in warm_start.Vs], # Xs
                warm_start.y, # y
                zero(T), # CX
                zeros(T, m), # AX
                warm_start.Vs, # Vs
                warm_start.mu, # mu
            )
        end
    end
end # Point

include("Printing.jl")
include("SdpData.jl")

# If something belongs to a specific block, the first index is always used to identify the block!
struct Data{
    T<:AbstractFloat,
    M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}}, # for constraints
    M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}}, # for cost matrices
}
    Bi_tildes::Vector{Vector{M}} # matrices Bi_tildes[b][i] where b is a block and i in [ns[b]]; each matrix Bi_tildes[b][i] is an (m x ns[i])-matrix
    Aj_iis::Vector{Vector{Vector{T}}} # vectors Aj_iis[b][i] where b is a block and i in [ns[b]] vector of vectors; each vector has length m and contains all entries Aj_ii
    AX_new::Vector{T} # to store the new value of A(X) during column updates; length m
    vi_diffs::Vector{Vector{T}} # vi_new - vi_old; vi_diffs[b] for block b
    V_T_vi_diffs::Vector{Vector{T}} # to store V^T * vi_diff; V_T_vi_diffs[b] for block b
    C_i_tildes::Vector{Vector{Vector{T}}} # vectors C_i_tildes[b][i] where b is a block and i in [ns[b]]; each vector has length ns[b] and contains column i of Cs[b] with a zero at position i
    b_minus_AX::Vector{T} # to store b - A(X) during column updates; length m
    minus_Bis::Vector{Vector{M}} # matrices -Bis[b][i] where b is a block and i in [ns[b]]; each matrix -Bis[b][i] is an (m x ns[i])-matrix
    y_plus_mu_b_minus_AX::Vector{T} # to store y + mu * (b - A(X)) during column updates; length m
    tmp_ns::Vector{Vector{T}} # vectors tmp_ns[b] where b is a block; each vector has length ns[b]
    vi_olds::Vector{Vector{T}} # vectors vi_olds[b] where b is a block; to store the previous column vi before updating
    tmp_vis::Vector{Vector{T}} # vectors tmp_vis[b] where b is a block; used during column updates
    my_indicess::Vector{Vector{Vector{Int}}} # vectors my_indicess[b][i] where b is a block and i in [ns[b]]; each vector contains all indices in which i (of block b) is involved
    tmp_my_indicess::Vector{Vector{Vector{T}}} # vectors tmp_my_indicess[b][i] where b is a block and i in [ns[b]]; each vector tmp_my_indicess[b][i] has length my_indicess[b][i]
    ineq_starts::Vector{Vector{Int64}} # vectors ineq_starts[b][i] where b is a block and i in [ns[b]]; each ineq_starts[b][i] contains the first index stored in my_indicess[b][i] that corresponds to an inequality

    function Data{T,M,M2}(
        sdp::SdpData{T,M,M2}, k::Int64
    ) where {
        T<:AbstractFloat,
        M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
        M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    }
        As::Vector{Vector{M}} = sdp.As
        As_vec::Vector{M} = sdp.As_vec
        ns::Vector{Int64} = sdp.ns
        Cs::Vector{M2} = sdp.Cs
        m::Int64 = sdp.m

        my_indicess::Vector{Vector{Vector{Int64}}} = [Vector{Int64}[] for _ in ns]
        ineq_starts::Vector{Vector{Int64}} = [zeros(Int64, n) for n in ns]
        Bi_tildes::Vector{Vector{M}} = [M[] for _ in ns]
        Aj_iis::Vector{Vector{Vector{T}}} = [Vector{T}[] for _ in ns]
        C_i_tildes::Vector{Vector{Vector{T}}} = [Vector{T}[] for _ in ns]
        minus_Bis::Vector{Vector{M}} = [M[] for _ in ns]

        for (b, n) in pairs(ns), i in 1:n
            A::Vector{M} = As[b]
            myInd::Vector{Int64} = []

            # compute/find "my_indices" for this column
            for (j, Ai) in pairs(A)
                if issparse(Ai)
                    # if any(!iszero, view(Ai, :, i))
                    if nnz(view(Ai, :, i)) > 0
                        push!(myInd, j)
                    end
                else
                    if any(Ai[:, i] .!= zero(T))
                        push!(myInd, j)
                    end
                end
            end
            push!(my_indicess[b], myInd)

            # find the first index corresponding to an inequality for this column
            ineq_starts[b][i] = length(my_indicess[b][i]) + 1 # default: no inequalities
            for j in eachindex(my_indicess[b][i])
                if my_indicess[b][i][j] >= sdp.index_ineq_start
                    ineq_starts[b][i] = j
                    break
                end
            end

            # create Bi_tilde for this column
            push!(Bi_tildes[b], As_vec[b][my_indicess[b][i], (1 + (i - 1) * n):(i * n)])
            Bi_tildes[b][i][:, i] .= zero(T)
            issparse(Bi_tildes[b][i]) && dropzeros!(Bi_tildes[b][i])
            @assert size(Bi_tildes[b][i]) == (length(my_indicess[b][i]), n)

            # create Aj_ii for this column
            push!(Aj_iis[b], As_vec[b][:, vecInd(n, i, i)])

            # create C_i_tilde for this column
            push!(C_i_tildes[b], Cs[b][1:n, i])
            C_i_tildes[b][i][i] = zero(T)
            issparse(C_i_tildes[b][i]) && dropzeros!(C_i_tildes[b][i])

            # create -Bi for this column
            push!(
                minus_Bis[b],
                transpose(-As_vec[b][my_indicess[b][i], vecInd(n, 1, i):vecInd(n, n, i)]),
            )
        end

        return new{T,M,M2}(
            Bi_tildes, # Bi_tildes
            Aj_iis, # Aj_iis
            zeros(T, m), # AX_new
            [zeros(T, min(n, k)) for n in ns], # vi_diffs
            [zeros(T, n) for n in ns], # V_T_vi_diffs
            C_i_tildes, # C_i_tildes
            zeros(T, m), # b_minus_AX
            minus_Bis, # minus_Bis
            zeros(T, m), # y_plus_mu_b_minus_AX
            [zeros(T, n) for n in ns], # tmp_ns
            [zeros(T, min(n, k)) for n in ns], # vi_olds
            [zeros(T, min(n, k)) for n in ns], # tmp_vis
            my_indicess, # my_indicess
            [
                Vector{T}[zeros(T, length(inds)) for inds in my_indicess[b]] for
                b in eachindex(my_indicess)
            ], # tmp_my_indicess
            ineq_starts, # ineq_starts
        )
    end
end # Data

function barvinok_pataki_bound(
    sdp::SdpData{T,M}
) where {T<:AbstractFloat,M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}}}
    m::Int64 = sdp.m
    n::Int64 = maximum(sdp.ns)
    return min(n, ceil(Int64, sqrt(2m)))
end

@inline function vecInd(n::Int64, i::Int64, j::Int64)
    @assert n >= 1
    @assert 1 <= i && i <= n
    @assert 1 <= j && j <= n
    return i + n * (j - 1)
end

# inverse of vecInd
@inline function indVec(n::Int64, ij::Int64)
    @assert 1 <= n && 1 <= ij && ij <= n * n
    j::Int64 = div(ij - 1, n) + 1
    i::Int64 = ij - n * (j - 1)
    @assert 1 <= i && i <= n
    @assert 1 <= j && j <= n
    return i, j
end

# function and gradient evaluation
# The funtion returns the function value and stores the gradient in "grad"
function f_and_g!(
    grad::Vector{T},
    vi::Vector{T},
    b::Int64,
    i::Int64,
    sdp::SdpData{T,M,M2},
    point::Point{T},
    vi_old::Vector{T},
    data::Data{T,M},
) where {
    T<:AbstractFloat,
    M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
}
    # When annotating the complexity, "m" always refers to length(data.my_indicess[b][i])
    my_indices::Vector{Int64} = data.my_indicess[b][i] # O(1)
    @views data.AX_new[my_indices] .= point.AX[my_indices] # O(m)
    norm_diff::T = dot(vi, vi) - dot(vi_old, vi_old) # O(k)
    @views data.AX_new[my_indices] .+= norm_diff .* data.Aj_iis[b][i][my_indices] # O(m)
    data.vi_diffs[b] .= vi .- vi_old # O(k)
    mul!(data.V_T_vi_diffs[b], transpose(point.Vs[b]), data.vi_diffs[b]) # O(k*n)
    tmp_m::Vector{T} = data.tmp_my_indicess[b][i] # O(1)
    @views tmp_m .= data.AX_new[my_indices] # O(m)
    mul!(tmp_m, data.Bi_tildes[b][i], data.V_T_vi_diffs[b], T(2.0), one(T)) # O(m*n)
    data.AX_new[my_indices] .= tmp_m # O(m)
    @views data.b_minus_AX[my_indices] .= sdp.b[my_indices] .- data.AX_new[my_indices] # O(m)
    @views data.y_plus_mu_b_minus_AX[my_indices] .= point.y[my_indices] # O(m)
    @views data.y_plus_mu_b_minus_AX[my_indices] .+= point.mu .* data.b_minus_AX[my_indices] # O(m)
    for ineq in data.ineq_starts[b][i]:length(my_indices) # O(m)
        if data.y_plus_mu_b_minus_AX[my_indices[ineq]] < zero(T) # O(1)
            data.y_plus_mu_b_minus_AX[my_indices[ineq]] = zero(T) # O(1)
            data.b_minus_AX[my_indices[ineq]] = zero(T) # O(1)
        end
    end
    @views tmp_m .= data.y_plus_mu_b_minus_AX[my_indices] # O(m)
    @inbounds mul!(data.tmp_ns[b], data.minus_Bis[b][i], tmp_m) # O(m*n)
    @views data.tmp_ns[b] .+= sdp.Cs[b][:, i] # O(n)
    point.Vs[b][:, i] .= vi # O(k)
    mul!(grad, point.Vs[b], data.tmp_ns[b]) # O(k*n)
    grad .*= T(2) # O(k)
    CX_new::T =
        point.CX +
        T(2) * dot(data.C_i_tildes[b][i], data.V_T_vi_diffs[b]) +
        sdp.Cs[b][i, i] * norm_diff # O(n)
    @views return CX_new +
                  T(0.5) *
                  point.mu *
                  dot(data.b_minus_AX[my_indices], data.b_minus_AX[my_indices]) +
                  dot(point.y[my_indices], data.b_minus_AX[my_indices]) # O(m)
end

function update_column!(
    point::Point{T},
    sdp::SdpData{T,M,M2},
    b::Int64,
    i::Int64,
    data::Data{T,M,M2},
    delta::T,
    epsilon::T,
    max_evals::Int64,
) where {
    T<:AbstractFloat,
    M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
}
    n::Int64 = sdp.ns[b]
    @assert 1 <= i && i <= n
    @assert delta > zero(T)
    @assert epsilon > zero(T)
    @assert max_evals > 0

    # Save current column vi to perform incremental updates/computations.
    @views data.vi_olds[b] .= point.Vs[b][:, i] # O(k)

    # Do first (function and) gradient evaluation to specify stopping criterion.
    f_and_g!(data.tmp_vis[b], data.vi_olds[b], b, i, sdp, point, data.vi_olds[b], data)

    # Set g_tol such that we always make some progress compared to the starting solution.
    g_tol::T = min(epsilon, delta * norm(data.tmp_vis[b], Inf))

    @views data.tmp_vis[b] .= data.vi_olds[b] # O(k)

    local evals::Int

    last_f::T = zero(T)
    function f(vi::Vector{T})
        return last_f
    end
    function grad!(grad::Vector{T}, vi::Vector{T})
        last_f = f_and_g!(grad, vi, b, i, sdp, point, data.vi_olds[b], data)
        return grad
    end

    res = Optim.optimize(
        f, # function
        grad!, # gradient
        data.tmp_vis[b], # starting point
        LBFGS(), # algorithm
        Optim.Options(;
            x_abstol=zero(T), # Absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
            x_reltol=zero(T), # Relative tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
            f_abstol=zero(T), # Absolute tolerance in changes of the objective value. Defaults to 0.0.
            f_reltol=zero(T), # Relative tolerance in changes of the objective value. Defaults to 0.0.
            g_abstol=g_tol, # Absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
            f_calls_limit=max_evals, # A soft upper limit on the number of objective calls. Defaults to 0 (unlimited).
            g_calls_limit=0, # A soft upper limit on the number of gradient calls. Defaults to 0 (unlimited).
            h_calls_limit=0, # A soft upper limit on the number of Hessian calls. Defaults to 0 (unlimited).
            allow_f_increases=true, # Allow steps that increase the objective value. Defaults to true. Note that, when this setting is true, the last iterate will be returned as the minimizer even if the objective increased.
            successive_f_tol=1, # Determines the number of times the objective is allowed to increase across iterations. Defaults to 1.
            iterations=max_evals, # How many iterations will run before the algorithm gives up? Defaults to 1_000.
            time_limit=NaN, # A soft upper limit on the total run time. Defaults to NaN (unlimited).
        ),
    )

    # check result
    if !Optim.converged(res)
        #@printf "The inner solver did not converge!\n"
    end
    @assert Optim.f_calls(res) == Optim.g_calls(res) # our implementation relies on this
    evals = Optim.g_calls(res)
    data.tmp_vis[b] .= res.minimizer

    @assert all(isfinite, data.tmp_vis[b])
    point.Vs[b][:, i] .= data.tmp_vis[b] # O(k)

    @views mul!(point.Xs[b][:, i], transpose(point.Vs[b]), data.tmp_vis[b]) # O(k*n)
    for j in 1:n
        point.Xs[b][i, j] = point.Xs[b][j, i]
    end

    my_indices::Vector{Int64} = data.my_indicess[b][i]

    # update point.AX and point.CX
    norm_diff::T =
        dot(data.tmp_vis[b], data.tmp_vis[b]) - dot(data.vi_olds[b], data.vi_olds[b]) # O(k)
    @views point.AX[my_indices] .+= norm_diff .* data.Aj_iis[b][i][my_indices] # O(m)
    data.vi_diffs[b] .= data.tmp_vis[b] .- data.vi_olds[b] # O(k)
    mul!(data.V_T_vi_diffs[b], transpose(point.Vs[b]), data.vi_diffs[b]) # O(k*n)

    tmp::Vector{T} = data.tmp_my_indicess[b][i]
    @views tmp .= point.AX[my_indices]

    mul!(tmp, data.Bi_tildes[b][i], data.V_T_vi_diffs[b], T(2.0), one(T)) # O(m*n)
    point.AX[my_indices] .= tmp

    point.CX +=
        T(2) * dot(data.C_i_tildes[b][i], data.V_T_vi_diffs[b]) +
        norm_diff * sdp.Cs[b][i, i] # O(n)

    return evals
end

"""
    Xs, y, Zs, status, ws = augmented_mixing(sdp::SdpData{T,M,M2};
                     tol::T = T(1e-12),
                     mu_start::T = T(sqrt(maximum(sdp.ns))),
                     time_limit::Float64 = typemax(Float64),
                     max_iters::Int = typemax(Int),
                     iters_Z::Int = 50,
                     scaling::Bool = true,
                     shuffling::Bool = false,
                     double_sweep::Bool = false,
                     warm_start::Union{WarmStart,Nothing} = nothing,
                     p::T = one(T),
                     delta::T = T(1e-2),
                     epsilon::T = T(1e-2),
                     max_evals::Int = 1000,
                     tau::T = T(1.03),
                     rat_min::T = T(0.8),
                     rat_max::T = T(1.2))

Solve the SDP provided in `sdp` using the Augmented Mixing Method.

# Arguments
- `sdp`: SDP problem data. See [`SdpData`](@ref).

# Optional keyword arguments
- `tol`: stopping tolerance
- `mu_start`: initial value of penalty parameter
- `time_limit`: maximum wall-clock time in seconds
- `max_iters`: maximum number of outer iterations
- `iters_Z`: frequency for computing psd projections
- `scaling`: whether automatic scaling is applied
- `shuffling`: whether column update order is randomized
- `double_sweep`: whether columns are updated in both forward and reverse order
- `warm_start`: can be specified for triggering a warm start
- `p`: dual step size
- `delta`: relative tolerance for solving subproblems
- `epsilon`: absolute tolerance for solving subproblems
- `max_evals`: maximum number of function and gradient evaluations per column update
- `tau`: factor for updating the penalty parameter
- `rat_min`: lower bound for ratio-based balancing of penalty parameter
- `rat_max`: upper bound for ratio-based balancing of penalty parameter

# Return values
- `Xs::Vector{Matrix{T}}`: primal matrices (symmetric and psd)
- `y::Vector{T}`: dual multipliers (nonnegative for inequalities)
- `Zs::Vector{Matrix{T}}`: dual slack matrices (symmetric and psd)
- `status::Symbol`: one of `:tol`, `:time`, `:iter` (reason for termination)
- `ws::WarmStart{T}`: struct to allow warm start from last iterate

# Notes
We refer to the preprint https://arxiv.org/abs/2507.20386 for implementation details and further explanations of all optional keyword arguments.
"""
function augmented_mixing(
    original_sdp::SdpData{T,M,M2};
    tol::T=T(1e-12),
    mu_start::T=T(sqrt(maximum(original_sdp.ns))),
    time_limit::Float64=typemax(Float64),
    max_iters::Int64=typemax(Int64),
    iters_Z::Int64=50,
    scaling::Bool=true,
    shuffling::Bool=false,
    double_sweep::Bool=false,
    warm_start::Union{WarmStart,Nothing}=nothing,
    p::T=one(T),
    delta::T=T(1e-2),
    epsilon::T=T(1e-2),
    max_evals::Int64=1000,
    tau::T=T(1.03),
    rat_min::T=T(0.8),
    rat_max::T=T(1.2),
) where {
    T<:AbstractFloat,
    M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
}
    @assert tol > zero(T)
    @assert time_limit > 0.0
    @assert max_iters > 0
    @assert iters_Z > 0
    @assert delta > zero(T)
    @assert epsilon > zero(T)
    @assert max_evals > 0
    @assert tau > one(T)
    @assert rat_min > zero(T)
    @assert rat_min < rat_max

    tstart::Float64 = time()

    print_dashed_line()
    @printf "coefficient ranges of original SDP:\n"
    print_coefficient_ranges(original_sdp)

    sdp::SdpData{T,M,M2}, scale_constraints::Vector{T}, scale_A::T, scale_b::T, scale_C::T, time_scaling::Float64 = scale_sdp(
        original_sdp, scaling
    ) # we work with this SDP internally
    if scaling
        @printf "\ncoefficient ranges after scaling:\n"
        print_coefficient_ranges(sdp)
    end

    (;
        As::Vector{Vector{M}},
        As_vec::Vector{M},
        b::Vector{T},
        Cs::Vector{Matrix{T}},
        m::Int64,
        ns::Vector{Int64},
        index_ineq_start::Int64,
    ) = sdp

    if M == Matrix{T}
        num_nnz::Int64 = sum(count(!iszero, A) for A in As_vec)
        @printf "\nThis is a dense SDP with %d nonzeros in A.\n" num_nnz
        @printf "Density of A: %g %%\n\n" 100.0 * num_nnz / (m * sum(n * n for n in ns))
    else
        num_nnz = sum(nnz, As_vec)
        @printf "\nThis is a sparse SDP with %d nonzeros in A.\n" num_nnz
        @printf "Density of A: %g %%\n\n" 100.0 * num_nnz / (m * sum(n * n for n in ns))
    end

    @printf "n = %d\n" sum(ns)
    if length(ns) > 1
        println("With block sizes $ns.")
    end
    @printf "m = %d\n" m
    @printf "equality constraints: %d\n" index_ineq_start - 1
    @printf "inequality constraints: %d\n" m - index_ineq_start + 1
    kk::Int64 = barvinok_pataki_bound(original_sdp)
    @assert kk > 0
    @printf "k = %d\n" kk

    kks::Vector{Int} = [min(n, kk) for n in ns]

    point::Point{T} = Point{T}(ns, kks, m, mu_start; warm_start)

    tstart_data::Float64 = time()
    data::Data{T,M,M2} = Data{T,M,M2}(sdp, kk)
    time_data = time() - tstart_data
    @printf "\nCreating \"data\" took %.2f seconds\n\n" time_data

    Aty_vecs::Vector{Vector{T}} = [zeros(T, n * n) for n in ns]
    Z_not_psds::Vector{Matrix{T}} = [zeros(T, n, n) for n in ns]
    Zs::Vector{Matrix{T}} = [zeros(T, n, n) for n in ns]
    Z_diffs::Vector{Matrix{T}} = [zeros(T, n, n) for n in ns]
    tmp_n_ns::Vector{Matrix{T}} = [zeros(T, n, n) for n in ns]
    tmp_m::Vector{T} = zeros(T, m)
    p_rat::Vector{T} = zeros(T, m)
    d_rat::Vector{T} = zeros(T, m)
    AX_old::Vector{T} = zeros(T, m)

    point.AX .= zero(T)
    point.CX = dot(Cs, point.Xs)
    for b in eachindex(As_vec, point.Xs)
        mul!(point.AX, As_vec[b], reshape(view(point.Xs[b], :, :), :), one(T), one(T))
    end

    total_sum_evals::Int64 = 0
    iter::Int64 = 0

    last_iter_evals::Vector{Int64} = zeros(Int64, 10)
    last_max_evals::Vector{Int64} = zeros(Int64, length(last_iter_evals))

    printStart()
    primal_obj, dual_obj, gap, pinf, dinf, compl, compl_not_psd_abs = compute_errors(
        sdp,
        point,
        Aty_vecs,
        Z_not_psds,
        Zs,
        Z_diffs,
        tmp_n_ns,
        tmp_m,
        0, # iter
        iters_Z,
        tol,
    )
    primal_obj *= scale_b * scale_C / scale_A
    dual_obj *= scale_b * scale_C / scale_A
    printUpdate(
        iter,
        time() - tstart,
        primal_obj,
        dual_obj,
        gap,
        pinf,
        dinf,
        compl,
        compl_not_psd_abs,
        point,
        false,
        total_sum_evals,
        last_iter_evals,
        last_max_evals,
        zero(T), # p_rat
        zero(T), # d_rat
        false, # double_sweep
    )

    status_code = :in_progress # will be :tol, :time, or :iter on termination

    pairs_list = [(i, n) for i in eachindex(ns) for n in 1:ns[i]]

    while true
        iter += 1

        @assert all(all.(isfinite, point.Xs))
        @assert all(all.(isfinite, point.Vs))

        AX_old .= point.AX # needed for computing d_res

        iter_evals::Int64 = 0
        max_iter_evals::Int64 = 0

        if shuffling
            shuffle!(pairs_list)
        end

        for (b, i) in pairs_list
            evals::Int64 = update_column!(point, sdp, b, i, data, delta, epsilon, max_evals)
            total_sum_evals += evals
            iter_evals += evals
            max_iter_evals = max(max_iter_evals, evals)
        end

        if double_sweep
            for (b, i) in reverse(pairs_list)
                evals::Int64 = update_column!(
                    point, sdp, b, i, data, delta, epsilon, max_evals
                )
                total_sum_evals += evals
                iter_evals += evals
                max_iter_evals = max(max_iter_evals, evals)
            end
        end

        last_iter_evals[iter % length(last_iter_evals) + 1] = iter_evals
        last_max_evals[iter % length(last_max_evals) + 1] = max_iter_evals

        # point.X always is up-to-date, i.e., we have point.X = point.V' * point.V
        # however, rounding errors might accumulate in point.CX and point.AX
        point.CX = dot(sdp.Cs, point.Xs)
        point.AX .= zero(T)
        for b in eachindex(As_vec, point.Xs)
            mul!(point.AX, As_vec[b], reshape(view(point.Xs[b], :, :), :), one(T), one(T))
        end

        primal_obj, dual_obj, gap, pinf, dinf, compl, compl_not_psd_abs = compute_errors(
            sdp,
            point,
            Aty_vecs,
            Z_not_psds,
            Zs,
            Z_diffs,
            tmp_n_ns,
            tmp_m,
            iter,
            iters_Z,
            tol,
        )

        success::Bool = pinf < tol && gap < tol && dinf < tol && compl < tol
        time_limit_reached::Bool = time() - tstart >= time_limit
        max_iters_reached::Bool = iter >= max_iters
        done::Bool = success || time_limit_reached || max_iters_reached
        if max_iters_reached
            status_code = :iter
        end
        if time_limit_reached
            status_code = :time
        end
        if success
            status_code = :tol
        end

        if iter % (15 * length(last_iter_evals)) == 0
            printStart()
        end

        p_rat .= b .- point.AX
        d_rat .= point.mu .* (point.AX .- AX_old)

        for ineq in (sdp.index_ineq_start):m
            if p_rat[ineq] < zero(T) && point.y[ineq] == zero(T)
                p_rat[ineq] = zero(T)
                d_rat[ineq] = zero(T)
            end
        end

        norm_p_rat::T = norm(p_rat)
        norm_d_rat::T = norm(d_rat)

        primal_obj *= scale_b * scale_C / scale_A
        dual_obj *= scale_b * scale_C / scale_A
        printUpdate(
            iter,
            time() - tstart,
            primal_obj,
            dual_obj,
            gap,
            pinf,
            dinf,
            compl,
            compl_not_psd_abs,
            point,
            done,
            total_sum_evals,
            last_iter_evals,
            last_max_evals,
            norm_p_rat,
            norm_d_rat,
            double_sweep,
        )

        if done
            if status_code == :time || status_code == :iter
                # compute Z to get the best possible return triple (X, y, Z)
                compute_errors(
                    sdp,
                    point,
                    Aty_vecs,
                    Z_not_psds,
                    Zs,
                    Z_diffs,
                    tmp_n_ns,
                    tmp_m,
                    0,
                    1,
                    tol,
                )
            end
            break
        end

        # update dual variables
        tmp_m .= b .- point.AX
        point.y .+= p .* point.mu .* tmp_m
        for ineq in (sdp.index_ineq_start):m
            point.y[ineq] = max(zero(T), point.y[ineq])
        end

        # update penalty parameter
        ratio::T = norm_p_rat / norm_d_rat
        if ratio < rat_min
            point.mu /= tau
        elseif ratio > rat_max
            point.mu *= tau
        end

        @assert point.mu > 0
        point.mu = max(point.mu, T(1e-8))
        point.mu = min(point.mu, T(1e8))
    end

    @assert status_code in [:tol, :time, :iter]

    print_dashed_line()

    # return approximate optimal solution X, y, Z
    for X in point.Xs
        X .*= scale_b / scale_A
    end
    tmp_m .= point.y
    tmp_m .*= scale_C / scale_A
    tmp_m ./= scale_constraints
    for Z in Zs
        Z .*= scale_C
    end

    print_dashed_line()

    warm_start::WarmStart{T} = WarmStart(point.mu, point.Vs, point.y)

    return point.Xs, tmp_m, Zs, status_code, warm_start
end

function compute_errors(
    sdp::SdpData{T,M,M2},
    point::Point{T},
    Atys::Vector{Vector{T}},
    Z_not_psds::Vector{Matrix{T}},
    Zs::Vector{Matrix{T}},
    Z_diffs::Vector{Matrix{T}},
    tmp_n_ns::Vector{Matrix{T}},
    tmp_m::Vector{T},
    iter::Int64,
    iters_Z::Int64,
    tol::T,
) where {
    T<:AbstractFloat,
    M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
}
    @assert iter >= 0
    @assert iters_Z >= 1
    @assert tol > zero(T)
    @assert all(all.(isfinite, point.Xs))
    @assert all(issymmetric, point.Xs)

    (;
        As::Vector{Vector{M}},
        As_vec::Vector{M},
        Cs::Vector{Matrix{T}},
        b::Vector{T},
        m::Int64,
        ns::Vector{Int64},
        index_ineq_start::Int64,
    ) = sdp

    primal_obj::T = dot(Cs, point.Xs)
    dual_obj::T = dot(b, point.y)
    gap::T = abs(primal_obj - dual_obj) / (one(T) + abs(primal_obj) + abs(dual_obj))

    #mul!(tmp_m, A, vec(point.X))
    # TODO make this better

    tmp_m .= 0
    for b in eachindex(As_vec, point.Xs)
        mul!(tmp_m, As_vec[b], reshape(view(point.Xs[b], :, :), :), one(T), one(T))
    end
    # mul!(tmp_m, A, reshape(view(point.X, :, :), :))

    tmp_m .*= -one(T)
    tmp_m .+= b
    for ineq in index_ineq_start:m
        tmp_m[ineq] = max(zero(T), tmp_m[ineq])
    end
    pinf::T = norm(tmp_m, Inf) / (one(T) + norm(b, Inf))

    for b in eachindex(Atys, As_vec)
        mul!(Atys[b], transpose(As_vec[b]), point.y)
    end
    #Z_not_psd .= Symmetric(C .- reshape(Aty, n, n))
    for b in eachindex(Z_not_psds, Cs, ns)
        Z_not_psds[b] .= Cs[b]
        n = ns[b]
        for i in 1:n, j in i:n
            Z_not_psds[b][i, j] -= Atys[b][i + n * (j - 1)]
            Z_not_psds[b][j, i] = Z_not_psds[b][i, j]
        end
        @assert issymmetric(Z_not_psds[b])
    end

    dot_X_Z_not_psd::T = dot(point.Xs, Z_not_psds)
    compl_not_psd_abs::T =
        (abs(dot_X_Z_not_psd)) / (one(T) + abs(primal_obj) + abs(dual_obj))

    dinf::T = typemax(T)
    compl::T = typemax(T)

    compute_Z::Bool =
        (iter % iters_Z == 0 && pinf < tol && gap < tol && compl_not_psd_abs < tol) ||
        (iters_Z == 1)

    # we do not compute Z after every iteration as this is quite expensive
    if compute_Z
        for b in eachindex(ns)
            n = ns[b]

            if T == Float64
                eigenvalues::Vector{T}, eigenvectors::Matrix{T} = LinearAlgebra.LAPACK.syevr!(
                    'V', 'A', 'U', Z_not_psds[b], 0.0, 0.0, 0, 0, 0.0
                )
            else
                F = eigen(Symmetric(Z_not_psds[b]))
                eigenvalues = F.values
                eigenvectors = F.vectors
            end

            #Z .= F.vectors * Diagonal(max.(F.values, zero(T))) * F.vectors'
            #fill!(tmp_n_n, zero(T))
            for i in 1:n, j in 1:n
                tmp_n_ns[b][i, j] = sqrt(max(zero(T), eigenvalues[j])) * eigenvectors[i, j]
            end
            mul!(Zs[b], tmp_n_ns[b], transpose(tmp_n_ns[b]))
            #@assert issymmetric(Zs[b])

            # to avoid cancellation when computing Z - Z_not_psd, directly compute Zdiff
            # Zdiff .= F.vectors * Diagonal(min.(F.values, zero(T))) * F.vectors'

            for i in 1:n
                for j in 1:n
                    tmp_n_ns[b][i, j] =
                        -sqrt(-min(zero(T), eigenvalues[j])) * eigenvectors[i, j]
                end
            end

            mul!(Z_diffs[b], tmp_n_ns[b], transpose(tmp_n_ns[b]))
            #@assert issymmetric(Z_diffs[b])
        end

        dinf = maximum(norm.(Z_diffs, Inf)) / (one(T) + maximum(norm.(Cs, Inf)))
        compl = abs(dot(point.Xs, Zs)) / (one(T) + abs(primal_obj) + abs(dual_obj))
    end

    @assert gap >= 0
    @assert pinf >= 0
    @assert dinf >= 0
    @assert compl >= 0
    @assert compl_not_psd_abs >= 0

    return primal_obj, dual_obj, gap, pinf, dinf, compl, compl_not_psd_abs
end

end # module
