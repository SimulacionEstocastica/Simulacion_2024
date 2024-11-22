module z2_lattice

using Agents
using IterTools
using CairoMakie # GLMakie permite interactividad
using StaticArrays
using LinearAlgebra


current_dir = @__DIR__

group_distributions = joinpath(current_dir, "..", "group_distributions")
U1_dist_path = joinpath(group_distributions, "U1_dist.jl")
Zn_dist_path = joinpath(group_distributions, "Zn_dist.jl")

include(U1_dist_path)
include(Zn_dist_path)

import .U1_dist: U1_boltzmann
import .Zn_dist: Zn_boltzmann

export initialize_model, Element, Edge, Vertex, Face

@agent struct Edge(GridAgent{2})
    angle::Float64
    energy_value::Float64
end

@agent struct Vertex(GridAgent{2})
    energy_value::Float64
end

@agent struct Face(GridAgent{2})
    energy_value::Float64
end

@multiagent Element(Edge, Vertex, Face) <: AbstractAgent

function update_value!(face::Element, model::StandardABM)
    @assert variant(face) isa Face
    β = model.β
    ordered_neighbor_ids = id_in_position.(SA[(0, -1).+face.pos, (1, 0).+face.pos, (0, 1).+face.pos, (-1, 0).+face.pos], [model])
    signs = SA[1, 1, -1, -1]
    face_angles = [model[i].angle * sign for (i, sign) in zip(ordered_neighbor_ids, signs)]
    face.energy_value = -cos(sum(face_angles))
end

function partial_face_angle(face::Element, edge::Element, model::StandardABM)
    @assert variant(face) isa Face && variant(edge) isa Edge
    ordered_neighbor_ids = id_in_position.(SA[(0, -1).+face.pos, (1, 0).+face.pos, (0, 1).+face.pos, (-1, 0).+face.pos], [model])
    signs = SA[1, 1, -1, -1]
    partial_face_angle = sum([sign * model[i].angle for (i, sign) in zip(ordered_neighbor_ids, signs) if model[i] != edge])
    edge_sign = signs[edge.id.==ordered_neighbor_ids][1]
    return partial_face_angle, edge_sign
end

"""
yang_mills_step!(agent, model)

Updates the angle of an edge agent by sampling from a Boltzmann distribution
conditional on the values of the edges in the faces that share the edge.

# Arguments
- `agent`: The edge agent to update.
- `model`: The model containing the agent.

# Returns
- `nothing`
"""
function yang_mills_step!(agent, model, ::Edge; distribution)::Nothing # Aquí podemos implementar multiple dispatch como en el tutorial (osea una que acepte ::Edge, otra que acepte ::Face, ::Vertex)
    β = model.β
    face_values::Vector{Float64} = []
    edge_signs::Vector{Int} = []
    for neighbour in nearby_agents(agent, model, 1)
        if variant(neighbour) isa Face
            partial_angle, edge_sign = partial_face_angle(neighbour, agent, model)
            push!(face_values, partial_angle)
            push!(edge_signs, edge_sign)
        end
    end
    agent.angle = distribution(face_values..., edge_signs[1], β)[1] # Aprovecho Multiple Dispatch. Slay!
    return nothing
end

function yang_mills_step!(agent, model, ::Face; distribution)::Nothing
    update_value!(agent, model)
    return nothing
end

function yang_mills_step!(agent, model, ::Vertex; distribution)::Nothing
    return nothing
end

yang_mills_step!(agent, model; distribution) = yang_mills_step!(agent, model, variant(agent); distribution)

"""
initialize_model(height::Int, width::Int, β::Float64)

Initializes a model with a grid of size `height` x `width` with a given β value.
This model is a 2D lattice with vertices, edges and faces. The vertices are initialized with no properties.
The edges are initialized with an angle of 1.0. The faces are initialized with a value of 1.0.

# Arguments
- `height::Int`: The height of the grid.
- `width::Int`: The width of the grid.
- `β::Float64`: The β value of the model.

# Returns
- `model`: The initialized model.
"""
function initialize_model(height::Int, width::Int, β::Float64, group::String)

    @assert occursin(r"^(U1|Z\d+)$", group) "Invalid group name. Must be 'U1' or 'Z' followed by a number."
	
    if group == "U1"
        distribution = U1_boltzmann
    elseif group[1] == 'Z'
        n = parse(Int, group[2:end])
        distribution = (kwargs...) -> Zn_boltzmann(kwargs...; n=n)
	else
		error("Invalid group name. Must be 'U1' or 'Z' followed by a number.")
    end

    gridsize::Tuple{Int,Int} = (2 * height - 1, 2 * width - 1)
    space = GridSpaceSingle(gridsize; periodic=false, metric=:manhattan)
    properties = Dict([:β => β, :dimensions => (height, width)])
    yang_mills_stepper! = (kwargs...) -> yang_mills_step!(kwargs...; distribution=distribution)
    model = StandardABM(
        Element,
        space;
        (agent_step!)=yang_mills_stepper!, properties,
        container=Vector,
        scheduler=Schedulers.ByID()
    ) # TODO: Ver cómo controlar mejor el Scheduler

    # We add the vertices
    for (i, j) in product(2 .* (1:height) .- 1, 2 .* (1:width) .- 1)
        add_agent!((i, j), constructor(Element, Vertex), model; energy_value=0.0)
    end

    # We add the edges initialized with angle 0.0
    edge_positions = Iterators.flatten(
        (
        product(2 .* (1:height) .- 1, 2 .* (1:width-1)),
        product(2 .* (1:height-1), 2 .* (1:width) .- 1),
    )
    )
    for (i, j) in edge_positions
        add_agent!((i, j), constructor(Element, Edge), model; angle=0.0, energy_value=0.0)
    end

    # We add the faces initialized with value 0.0
    for (i, j) in product(2 .* (1:height-1), 2 .* (1:width-1))
        add_agent!((i, j), constructor(Element, Face), model; energy_value=0.0)
    end

    return model
end



main = false
if main
    # Available groups: "U1", "Zn" (n=1, 2, 3, 4, 5, 6, ...)

    group = "Z2"
    height = 10
    width = 10
    β = 1.0

    model = initialize_model(width, height, β, group)

    step!(model, 100)

    # Plotting
    agent_color(agent) = variant(agent) isa Edge ? "#FF0000" : variant(agent) isa Vertex ? "#0000FF" : :yellow
    fig, ax, _ = abmplot(model; agent_color=agent_color)
    arrows!(ax,
        [Point2f(edge.pos...) for edge in allagents(model) if variant(edge) isa Edge],
        [Vec2f(cos(edge.angle), sin(edge.angle)) for edge in allagents(model) if variant(edge) isa Edge]
    )

	
	if group == "U1"
		bins = 100
	elseif group[1] == 'Z'
		n = parse(Int, group[2:end])
		bins = n
	end

    ax2 = Axis(fig[1, 2])
    hist!(ax2, [face.energy_value for face in allagents(model) if variant(face) isa Face], bins=bins, normalization=:pdf)

    window = display(fig)
end

# TODO: Hay que ver si esto tiene sentido. Y visualizarlo mejor! Pero en otro archivo...

end