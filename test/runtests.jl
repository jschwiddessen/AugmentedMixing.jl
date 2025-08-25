using AugmentedMixing
using Test

using LinearAlgebra, SparseArrays, DoubleFloats, Random

Random.seed!(12345)

@inline function vecInd(n::Int64, i::Int64, j::Int64)
    @assert n >= 1
    @assert 1 <= i && i <= n
    @assert 1 <= j && j <= n
    return i + n * (j - 1)
end

function random_qubo_instance(n, cuts::Bool, T)
    C::Matrix{Float64} = Symmetric(rand(Float64, n, n))

    num_cuts::Int64 = 4 * n * (n - 1) * (n - 2) / 6
    m::Int64 = cuts ? n + num_cuts : n
    b::Vector{T} = cuts ? [ones(T, n); -ones(T, num_cuts)] : ones(T, n)

    I::Vector{Int64} = []
    J::Vector{Int64} = []
    V::Vector{T} = []

    sizehint!(I, cuts ? n + 6 * num_cuts : n)
    sizehint!(J, cuts ? n + 6 * num_cuts : n)
    sizehint!(V, cuts ? n + 6 * num_cuts : n)

    for i in 1:n
        push!(I, i)
        push!(J, vecInd(n, i, i))
        push!(V, one(T))
    end

    if cuts
        next::Int64 = n + 1
        for i in 1:n
            for j in (i + 1):n
                for k in (j + 1):n
                    push!(I, next)
                    push!(J, vecInd(n, i, j))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, i))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, i, k))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, i))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, k))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, j))
                    push!(V, T(0.5))
                    next += 1

                    push!(I, next)
                    push!(J, vecInd(n, i, j))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, i))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, i, k))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, i))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, k))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, j))
                    push!(V, -T(0.5))
                    next += 1

                    push!(I, next)
                    push!(J, vecInd(n, i, j))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, i))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, i, k))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, i))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, k))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, j))
                    push!(V, -T(0.5))
                    next += 1

                    push!(I, next)
                    push!(J, vecInd(n, i, j))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, i))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, i, k))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, i))
                    push!(V, -T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, j, k))
                    push!(V, T(0.5))
                    push!(I, next)
                    push!(J, vecInd(n, k, j))
                    push!(V, T(0.5))
                    next += 1
                end
            end
        end
    end

    return SdpData(SparseArrays.sparse!(I, J, V, m, n * n), b, -C, n + 1)
end

@testset "AugmentedMixing.jl" begin
    n = 5
    sdp = random_qubo_instance(n, true, Float64)
    Xs, y, Zs, status, ws = augmented_mixing(sdp; max_iters=100_000)
    @test status == :tol

    # use DoubleFloats
    Xs, y, Zs, status, ws = augmented_mixing(
        SdpData(sdp, Double64);
        tol=Double64(1e-20),
        max_iters=100_000,
        warm_start=WarmStart(ws, Double64),
    )
    @test status == :tol
end
