module U1_dist

using ApproxFun
using StaticArrays

export U1_boltzmann

"""
U1_boltzmann(
    first_partial_value::Float64,
    second_partial_value::Float64,
    edge_sign::Int,
    β::Float64;
    n_samples::Int=1
)::Vector{Float64}

Computes the Yang-Mills Boltzmann distribution for a given set of partial values and a β value.

# Arguments
- `first_partial_value::Float64`: The partial value associated with the first face.
- `second_partial_value::Float64`: The partial value associated with the second face.
- `edge_sign::Int`: The sign of the edge.
- `β::Float64`: The β value of the distribution.
- `n_samples::Int=1`: The number of samples to draw from the distribution.

# Returns
- `Vector{Float64}`: The samples drawn from the distribution.
"""
function U1_boltzmann(
    first_partial_value::Float64,
    second_partial_value::Float64,
    edge_sign::Int,
    β;
    n_samples::Int=1
)::Vector{Float64}

    pdf = Fun(
        θ -> exp.(β .* (cos.(first_partial_value .+ (edge_sign .* θ)) .+ cos.(second_partial_value .- (edge_sign .* θ)))),
        Interval(0.0, 2π)
    )
    samples = sample(pdf, n_samples)

    return samples
end

"""
U1_boltzmann(
    only_partial_value::Float64,
    edge_sign::Int,
    β::Float64;
    n_samples::Int=1
)::Vector{Float64}

Computes the Yang-Mills Boltzmann distribution for a given partial value and a β value.
Used for faces with only one neighbouring face.

# Arguments
- `only_partial_value::Float64`: The partial value associated with the face.
- `edge_sign::Int`: The sign of the edge.
- `β::Float64`: The β value of the distribution.
- `n_samples::Int=1`: The number of samples to draw from the distribution.

# Returns
- `Vector{Float64}`: The samples drawn from the distribution.
"""
function U1_boltzmann(
    only_partial_value::Float64,
    edge_sign::Int,
    β; n_samples::Int=1
)::Vector{Float64}

    pdf = Fun(
        θ -> exp.(β .* cos.(only_partial_value .+ (edge_sign .* θ))),
        Interval(0.0, 2π)
    )
    samples = sample(pdf, n_samples)

    return samples
end

main = false

if main
    using GLMakie
    samples = U1_boltzmann(pi / 2, -1, 10.0, n_samples=10^6)
    nomralized_samples = samples ./ 2π
    println("max: ", maximum(nomralized_samples))
    println("min: ", minimum(nomralized_samples))
    using Statistics
    println("mean: ", mean(nomralized_samples))
    hist(samples, bins=100, normalization=:pdf)
end

end