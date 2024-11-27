using ProgressMeter
using GLMakie

plotFontsize = 30

current_dir = @__DIR__

z2_visuals_path = joinpath(current_dir, "src", "ambient_space", "z2_visuals.jl")
z2_wilson_path = joinpath(current_dir, "src", "observables", "wilson_loop", "Z2_wilson.jl")

include(z2_visuals_path)
include(z2_wilson_path)

import .z2_visuals: interactive_exploration
import .z2_Wilson_Loop: get_V_func14

# Available groups: "U1", "Zn" (n=1, 2, 3, 4, 5, 6, ...)

group = "U1"
height = 10
width = 10
β = 10.0
interactive_exploration(height, width , β, group)

"""
group = "Z2"
height = 20
width = 20
β = 5.0
n_samples = 1000
V_β = get_V_func14(width, height, β, group, n_samples)

# we plot Re(V), Im(V) and |V| for R_ in {1,2,..,height-1}
R_::Vector{Int} = 1:height-1

V_R_progress = Progress(length(R_), 1, "Computing V(R_)...")
_V_R_::Vector{Tuple{Int, Complex{Float64}}} = []
for r in R_
    push!(_V_R_, (r, V_β(r)))
    next!(V_R_progress)
end

# Extract the complex values from the tuples
complex_values = [x[2] for x in _V_R_]  # Create an array of ComplexF64 from tuples

# Now you can work with complex_values (which is a Vector{ComplexF64})
real_part = real.(complex_values)  # Real parts of the complex values
imaginary_part = imag.(complex_values)  # Imaginary parts of the complex values
norms = abs.(complex_values)  # Norm (magnitude) of the complex values

# Compute the ratio (x, y/x) for each plot
x_values = [x[1] for x in _V_R_]  # Extract x (the first element) from each tuple
real_part_ratio = real_part ./ x_values  # y/x for real part
imaginary_part_ratio = imaginary_part ./ x_values  # y/x for imaginary part
norms_ratio = norms ./ x_values  # y/x for norms

# Create the plots for the real part
fig1 = Figure(fontsize = plotFontsize)
ax1 = Axis(fig1[1, 1], title = "Real Part", xlabel = "R", ylabel = "Real")
lines!(ax1, 1:length(real_part), real_part, label = "Real Part")

fig2 = Figure(fontsize = plotFontsize)
ax2 = Axis(fig2[1, 1], title = "Real Part / R", xlabel = "R", ylabel = "Real / R")
lines!(ax2, 1:length(real_part_ratio), real_part_ratio, label = "Real Part / R")

# Create the plots for the imaginary part
fig3 = Figure(fontsize = plotFontsize)
ax3 = Axis(fig3[1, 1], title = "Imaginary Part", xlabel = "R", ylabel = "Imaginary")
lines!(ax3, 1:length(imaginary_part), imaginary_part, label = "Imaginary Part")

fig4 = Figure(fontsize = plotFontsize)
ax4 = Axis(fig4[1, 1], title = "Imaginary Part / R", xlabel = "R", ylabel = "Imaginary / R")
lines!(ax4, 1:length(imaginary_part_ratio), imaginary_part_ratio, label = "Imaginary Part / R")

# Create the plots for the norm
fig5 = Figure(fontsize = plotFontsize)
ax5 = Axis(fig5[1, 1], title = "Norm", xlabel = "R", ylabel = "Norm")
lines!(ax5, 1:length(norms), norms, label = "Norm")

fig6 = Figure(fontsize = plotFontsize)
ax6 = Axis(fig6[1, 1], title = "Norm / R", xlabel = "R", ylabel = "Norm / R")
lines!(ax6, 1:length(norms_ratio), norms_ratio, label = "Norm / R")

# 7th plot: Colored scatter of the data (real part vs imaginary part) connected by a line
fig7 = Figure(fontsize = plotFontsize)
ax7 = Axis(fig7[1, 1], title = "Colored Scatter of Data", xlabel = "Real Part", ylabel = "Imaginary Part")
scatter!(ax7, real_part, imaginary_part, color = 1:length(real_part), colormap = :viridis)
lines!(ax7, real_part, imaginary_part, color = 1:length(real_part), colormap = :viridis)

# 8th plot: Colored scatter of the data divided by R (real part / R vs imaginary part / R) connected by a line
fig8 = Figure(fontsize = plotFontsize)
ax8 = Axis(fig8[1, 1], title = "Colored Scatter of Data / R", xlabel = "Real Part / R", ylabel = "Imaginary Part / R")
scatter!(ax8, real_part_ratio, imaginary_part_ratio, color = 1:length(real_part_ratio), colormap = :viridis)
lines!(ax8, real_part_ratio, imaginary_part_ratio, color = 1:length(real_part_ratio), colormap = :viridis)

# Ensure the 'data' directory exists
mkpath("data")
mkpath("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)")

# Save the plots to the 'data' directory
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/Real Part.svg", fig1)
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/Real Part div R.svg", fig2)
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/Img Part.svg", fig3)
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/Img Part div R.svg", fig4)
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/Norm.svg", fig5)
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/Norm div R.svg", fig6)
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/All.svg", fig7)
save("data/$(height) x $(width) x $(β) x $(group) x $(n_samples)/All div R.svg", fig8)

println("Plots saved to the 'data' directory.")
""";
