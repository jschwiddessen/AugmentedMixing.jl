# SDP input format

using SparseArrays

export SdpData

"""
    struct SdpData{
        T<:AbstractFloat,
        M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
        M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    }

Used to specify an SDP that can be solved with [`augmented_mixing`](@ref).

An SDP with q blocks reads:

minimize `< C_1 , X_1 > + ... + < C_q , X_q >` subject to `A_1(X_1) + ... + A_q(X_q) = (or >=) b` and `X_1, ... , X_q` are symmetric, positive semidefinite matrices

# Notes
All provided matrices must be symmetric. All matrices and vectors must be either of type Matrix{T} or SparseMatrixCSC{T,Int64}.

Inequality constraints are supported. All affine constraints must be ordered such that all equality constraints appear first,
followed by all inequality constraints. The argument `index_ineq_start` must be set to the index of the first inequality constraint.
If there are no inequality constraints, then `index_ineq_start` must be set to length(b) + 1.

The constraint matrices of affine constraints can be specified in two ways:

1) A vector of symmetric matrices (same length as b). For single-block SDPs, the corresponding container is `A::Vector{M}` and for multi-block SDPs, the container is `As::Vector{Vector{M}}`.

2) A single big matrix in which each row contains the respective vectorized constraint matrix. For single-block SDPs, the corresponding container is `A::M` and for multi-block SDPs, the container is `As_vec::Vector{M}`.

For containers with two dimensions, the first dimension always identifies the block.

# Constructors

General constructors for multi-block SDPs:

    sdp = SdpData(As::Vector{Vector{M}}, b::Vector{T}, Cs::Vector{M2}, index_ineq_start::Int64)
    sdp = SdpData(As_vec::Vector{M}, b::Vector{T}, Cs::Vector{M2}, index_ineq_start::Int64)

Constructors for single-block SDPs:

    sdp = SdpData(A::M, b::Vector{T}, C::M2, index_ineq_start::Int64)
    sdp = SdpData(A::Vector{M}, b::Vector{T}, C::M2, index_ineq_start::Int64)
"""
struct SdpData{
    T<:AbstractFloat,
    M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
}
    As::Vector{Vector{M}}
    As_vec::Vector{M}
    b::Vector{T}
    Cs::Vector{M2}
    m::Int64
    ns::Vector{Int64}
    index_ineq_start::Int64

    function SdpData(
        As::Vector{Vector{M}}, b::Vector{T}, Cs::Vector{M2}, index_ineq_start::Int64
    ) where {
        T<:AbstractFloat,
        M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
        M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    }
        @assert length(As) >= 1 && length(As) == length(Cs)
        @assert all(length(A) >= 1 for A in As)
        @assert all(all(size(Ai) .>= (1, 1)) for A in As for Ai in A)
        for A in As
            @assert (all(issparse(Ai) for Ai in A) || all(!issparse(Ai) for Ai in A))
        end

        m::Int64 = size(As[1], 1)
        ns::Vector{Int64} = [size(C, 1) for C in Cs]

        @assert all(ns .>= 1)
        @assert 1 <= index_ineq_start && index_ineq_start <= m + 1
        @assert m == length(b)
        @assert all(m == length(A) for A in As)

        @assert all((ns[i], ns[i]) == size(Cs[i]) for i in eachindex(ns, Cs))
        @assert all((ns[i], ns[i]) == size(Ai) for i in eachindex(As) for Ai in As[i])

        @assert all(issymmetric(C) for C in Cs)
        @assert all(issymmetric(Ai) for A in As for Ai in A)

        # construct As_vec
        As_vec::Vector{M} = []
        sizehint!(As_vec, length(ns))
        for i in eachindex(ns, As)
            A::Vector{M} = As[i]
            if issparse(A[1]) # sparse matrices
                I::Vector{Int64} = []
                J::Vector{Int64} = []
                V::Vector{T} = []
                for j in 1:m
                    Aj_I::Vector{Int64}, Aj_J::Vector{Int64}, Aj_V::Vector{T} = findnz(A[j])
                    for k in eachindex(Aj_I, Aj_J, Aj_V)
                        push!(I, j)
                        push!(J, vecInd(ns[i], Aj_I[k], Aj_J[k]))
                        push!(V, Aj_V[k])
                    end
                end
                push!(As_vec, SparseArrays.sparse!(I, J, V, m, ns[i] * ns[i]))
            else # dense matrices
                my_A::M = zeros(T, m, ns[i] * ns[i])
                for j in 1:m
                    # my_A[j, :] .= vec(A[j])
                    my_A[j, :] .= reshape(view(A[j], :, :), :)
                end
                push!(As_vec, my_A)
            end
        end
        return new{T,M,M2}(As, As_vec, b, Cs, m, ns, index_ineq_start)
    end

    function SdpData(
        A::Vector{M}, b::Vector{T}, C::M2, index_ineq_start::Int64
    ) where {
        T<:AbstractFloat,
        M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
        M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    }
        return SdpData([A], b, [C], index_ineq_start)
    end

    function SdpData(
        As_vec::Vector{M}, b::Vector{T}, Cs::Vector{M2}, index_ineq_start::Int64
    ) where {
        T<:AbstractFloat,
        M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
        M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    }
        m::Int64 = size(As_vec[1], 1)
        ns::Vector{Int64} = [size(C, 1) for C in Cs]

        @assert all(ns .>= 1)
        @assert all(m == size(A, 1) for A in As_vec)
        @assert 1 <= index_ineq_start && index_ineq_start <= m + 1
        @assert all((m, ns[i] * ns[i]) == size(As_vec[i]) for i in eachindex(ns, As_vec))
        @assert m == length(b)
        @assert all((ns[i], ns[i]) == size(Cs[i]) for i in eachindex(ns, Cs))
        @assert all(issymmetric(C) for C in Cs)

        As::Vector{Vector{M}} = []
        sizehint!(As, length(ns))
        for i in eachindex(ns)
            n::Int64 = ns[i]
            if issparse(As_vec[i])
                I::Vector{Int64}, J::Vector{Int64}, V::Vector{T} = findnz(As_vec[i])
                p = sortperm(I)
                permute!(I, p)
                permute!(J, p)
                permute!(V, p)
                @assert issorted(I)

                A::Vector{M} = []
                sizehint!(A, m)
                index::Int64 = 1
                for j in 1:m
                    I_A::Vector{Int64} = []
                    J_A::Vector{Int64} = []
                    V_A::Vector{T} = []
                    while (index <= length(I) && I[index] == j)
                        x, y = indVec(n, J[index])
                        push!(I_A, x)
                        push!(J_A, y)
                        push!(V_A, V[index])
                        index += 1
                    end
                    push!(A, SparseArrays.sparse!(I_A, J_A, V_A, n, n))
                end
                push!(As, A)
            else # dense matrices
                push!(
                    As, [reshape(view(As_vec[i], j, :), n, n) for j in 1:size(As_vec[i], 1)]
                )
            end
        end

        @assert all(issymmetric(Ai) for A in As for Ai in A)
        return new{T,M,M2}(As, As_vec, b, Cs, m, ns, index_ineq_start)
    end

    function SdpData(
        A::M, b::Vector{T}, C::M2, index_ineq_start::Int64
    ) where {
        T<:AbstractFloat,
        M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
        M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    }
        return SdpData([A], b, [C], index_ineq_start)
    end

    function SdpData(
        sdp::SdpData{T,M,M2}, T_new::Type{<:AbstractFloat}
    ) where {
        T<:AbstractFloat,
        M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
        M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    }
        return SdpData(
            map.(T_new, sdp.As_vec),
            map(T_new, sdp.b),
            map.(T_new, sdp.Cs),
            sdp.index_ineq_start,
        )
    end
end

function scale_sdp(
    sdp::SdpData{T,M,M2}, use_scaling::Bool
) where {
    T<:AbstractFloat,
    M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
    M2<:Union{Matrix{T},SparseMatrixCSC{T,Int64}},
}
    tstart::Float64 = time()

    #A s::Vector{Vector{M}} = deepcopy(sdp.As)
    As::Vector{Vector{M}} = map(x -> copy.(x), sdp.As)

    b::Vector{T} = deepcopy(sdp.b)
    Cs::Vector{M2} = deepcopy(sdp.Cs)

    # drop zeros in sparse matrices
    for j in 1:(sdp.m)
        for i in eachindex(As)
            issparse(As[i][j]) && dropzeros!(As[i][j])
        end
    end

    scale_constraints::Vector{T} = ones(T, sdp.m)

    if use_scaling
        # scale each constraint separately
        for i in eachindex(b)

            # number_of_cols = sum(count(x->nnz(x)>0, eachcol(A[i])) for A in As) 

            #scale_constraints[i] = maximum(norm(A[i]) for A in As)
            scale_constraints[i] = norm([norm(A[i]) for A in As]) #* sqrt(number_of_cols)
            b[i] /= scale_constraints[i]
            for A in As
                if issparse(A[i])
                    A[i].nzval ./= scale_constraints[i]
                else
                    A[i] ./= scale_constraints[i]
                end
            end
        end
    end

    scale_A::T = use_scaling ? maximum(norm(Ai) for A in As for Ai in A) : one(T) # TODO parallelize?
    #scale_A::T = use_scaling ? norm([norm(Ai) for A in As for Ai in A]) : one(T) # TODO parallelize?
    scale_b::T = use_scaling ? norm(b) : one(T)
    # scale_b::T = use_scaling ? scale_A : one(T)
    #scale_C::T = use_scaling ? maximum(norm(C) for C in Cs) : one(T)
    scale_C::T = use_scaling ? norm([norm(C) for C in Cs]) : one(T)
    if iszero(scale_C)
        scale_C = one(T)
    end

    if use_scaling
        for j in 1:(sdp.m)
            for i in eachindex(As)
                if issparse(As[i][j])
                    As[i][j].nzval ./= scale_A
                else
                    As[i][j] ./= scale_A
                end
            end
        end
        b ./= scale_b
        Cs ./= scale_C
    end

    @assert length(scale_constraints) == sdp.m
    @assert minimum(scale_constraints) > zero(T)
    @assert scale_A > zero(T)
    @assert scale_b > zero(T)
    @assert scale_C > zero(T)

    time_scaling::Float64 = time() - tstart

    if use_scaling
        @printf "\nCreating the scaled SDP took %.2f seconds\n" time_scaling
    end

    return SdpData(As, b, Cs, sdp.index_ineq_start),
    scale_constraints, scale_A, scale_b, scale_C,
    time_scaling
end

function print_coefficient_ranges(
    sdp::SdpData{T,M}
) where {T<:AbstractFloat,M<:Union{Matrix{T},SparseMatrixCSC{T,Int64}}}
    min_val::T, max_val::T = coefficient_range(sdp.As)
    @printf "A range: [%.0e, %.0e]\n" min_val max_val
    min_val, max_val = coefficient_range(sdp.b)
    @printf "b range: [%.0e, %.0e]\n" min_val max_val
    min_val, max_val = coefficient_range(sdp.Cs)
    @printf "C range: [%.0e, %.0e]\n" min_val max_val
end

function coefficient_range(As::Vector{Vector{Matrix{T}}}) where {T<:Real}
    min_val, max_val = typemax(T), zero(T)
    for A in As
        for Ai in A
            for x in Ai
                ax = abs(x)
                if ax > zero(T)
                    min_val = min(min_val, ax)
                    max_val = max(max_val, ax)
                end
            end
        end
    end
    return min_val, max_val
end

function coefficient_range(As::Vector{Vector{SparseMatrixCSC{T,Int64}}}) where {T<:Real}
    min_val, max_val = typemax(T), zero(T)
    for A in As
        for Ai in A
            for x in nonzeros(Ai)
                ax = abs(x)
                if ax > zero(T)
                    min_val = min(min_val, ax)
                    max_val = max(max_val, ax)
                end
            end
        end
    end
    return min_val, max_val
end

function coefficient_range(b::Vector{T}) where {T<:Real}
    min_val, max_val = typemax(T), zero(T)
    for x in b
        ax = abs(x)
        if ax > zero(T)
            min_val = min(min_val, ax)
            max_val = max(max_val, ax)
        end
    end
    return min_val, max_val
end

function coefficient_range(Cs::Vector{Matrix{T}}) where {T<:Real}
    min_val, max_val = typemax(T), zero(T)
    for C in Cs
        for x in C
            ax = abs(x)
            if ax > zero(T)
                min_val = min(min_val, ax)
                max_val = max(max_val, ax)
            end
        end
    end
    return min_val, max_val
end

function coefficient_range(Cs::Vector{SparseMatrixCSC{T,Int64}}) where {T<:Real}
    min_val, max_val = typemax(T), zero(T)
    for C in Cs
        for x in nonzeros(C)
            ax = abs(x)
            if ax > zero(T)
                min_val = min(min_val, ax)
                max_val = max(max_val, ax)
            end
        end
    end
    return min_val, max_val
end
