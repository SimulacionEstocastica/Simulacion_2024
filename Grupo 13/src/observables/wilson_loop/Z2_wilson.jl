"""
The objective of this module is to develope functionalities for Wilson Loops analysis.
"""

module z2_Wilson_Loop

using Agents
using Statistics

current_dir = @__DIR__

z2_lattice_path = joinpath("..", "..", "ambient_space", "z2_lattice.jl")

include(z2_lattice_path)

import .z2_lattice: initialize_model, Element, Edge, Vertex, Face

export get_V_func14

"""
W_(gamma::Array)::Complex{Float64}

Computes the Wilson Loop for a given path gamma.

# Arguments
- `gamma::Array`: The path for which the Wilson Loop will be computed. 
An array of tuples with an edge agent an a direction (1 for positive, -1 for negative. Up and right are possitive).

# Returns
- `Float64`: The Wilson Loop for the given path.
"""
function W_(γ::Array{Tuple{Element, Int}, 1})::Complex{Float64}
    loop = sum(direction * edge.angle for (edge, direction) in γ)
    return exp(im * loop)
end

"""
V(model:: StandardABM, R:: int)::Float64

Computes the potential between a static quark and antiquark separated by distance R.

# Arguments
- `model:: StandardABM`: The model in which the potential will be computed.
- `R:: int`: The distance between the quark and antiquark.

# Returns
- `Float64`: The potential between the quark and antiquark separated by distance R.


We use the formula

V(R) = - lim_{T → ∞} (1/T) log⟨W(γ_{R,T})⟩ 

where γ_{R,T} is a path of length T and breadth R and ⟨ ⋅ ⟩ denotes the expectation value.
"""
function V(model::StandardABM, R::Int, n_samples::Int = 1000)::Complex{Float64}
    h_vertices, v_vertices = model.dimensions
    T::Int = h_vertices - 1

    # we reach equilibrium
    step!(model, 100)  # NO GUARANTEE OF CONVERGENCE

    # Use sampling from pre-allocated memory
    samples = 2 .* rand(1:v_vertices - R, n_samples) .- 1
    bottom_left_vertices = hcat(ones(Int, n_samples), samples)

    # Pre-allocate wilson_loops array
    wilson_loops = Complex{Float64}[]

    # Main loop over bottom-left vertices
    for bottom_left in eachrow(bottom_left_vertices)
        step!(model, 1)  # this may not be necessary for each sample

        # Pre-allocate γ array for this particular sample
        γ = Vector{Tuple{Element, Int}}(undef, 2 * R + 2 * T)
        position = bottom_left

        # Create Wilson loop path
        idx = 1
        # First loop of length T
        for _ in 1:T
            position += [2, 0]
            ID = id_in_position(position - [1, 0], model)
            edge = model[ID]
            γ[idx] = (edge, 1)
            idx += 1
        end
        # Second loop of length R
        for _ in 1:R
            position += [0, 2]
            ID = id_in_position(position - [0, 1], model)
            edge = model[ID]
            γ[idx] = (edge, 1)
            idx += 1
        end
        # Third loop of length T
        for _ in 1:T
            position -= [2, 0]
            ID = id_in_position(position + [1, 0], model)
            edge = model[ID]
            γ[idx] = (edge, -1)
            idx += 1
        end
        # Fourth loop of length R
        for _ in 1:R
            position -= [0, 2]
            ID = id_in_position(position + [0, 1], model)
            edge = model[ID]
            γ[idx] = (edge, -1)
            idx += 1
        end

        # Compute the Wilson loop for this sample
        push!(wilson_loops, W_(γ))
    end

    # Calculate the mean and return the result
    E_Wγ = mean(wilson_loops)
    V_R = -log(E_Wγ) / T

    return V_R
end

"""
get_V_func14(width:: Int, height::Int, β::Float64, group::String, n_samples::Int)::Function

Returns a function that computes the potential between a static quark and antiquark separated by distance R.

# Arguments
- `width:: Int`: The width of the grid.
- `height:: Int`: The height of the grid.
- `β:: Float64`: The β value of the model.
- `group:: String`: The group of the model. Available groups: "U1", "Zn" (n=1, 2, 3, 4, 5, 6, ...)
- `n_samples:: Int`: The number of samples used to compute the potential.

# Returns
- `Function`: A function that computes the potential between a static quark and antiquark separated by distance R.
"""
function get_V_func14(width:: Int, height::Int, β::Float64, group::String, n_samples::Int)::Function
    model = initialize_model(width, height, β, group)
    function V_β(R::Int)::Complex{Float64}
        return V(model, R, n_samples)
    end
    
    return V_β
end

end

