import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import os
cwd = os.getcwd()

# El código de esta función se hizo usando Copilot

def animSLE(G, D, A):
    """
    Creates an animation showing G in black, D in red, and updates showing
    the vertices of A in order as green dots.

    Parameters:
    G (numpy.ndarray): Matrix representing vertices of rescalated ST.
    D (numpy.ndarray): Matrix representing vertices of the dual tree.
    A (list): List of indices indicating the order to update vertices.
    """
    cmap = mcolors.ListedColormap(['white', 'purple', 'green', 'orange'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    im_plot = 2*D + G
    # ax.plot(G[:, 0], G[:, 1], 'ks', ms=5, label='Spanning Tree')
    # ax.plot(D[:, 0], D[:, 1], 'rs', ms=5, label='Grafo Dual')
    # green_dots, = ax.plot([], [], 'go', lw=3, linestyle='-', label='Curva')
    ax.set_title('Curva $SLE_8$')
    im = ax.imshow(im_plot, cmap=cmap, norm=norm)
    # def init():
    #     green_dots.set_data([], [])
    #     return green_dots,

    def update(frame):
        x, y = A[frame]
        im_plot[x, y] = 3
        im.set_data(im_plot)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(A), blit=True, interval=20/len(A)*1000, repeat=False)
    # ax.legend()
    # video = ani.to_html5_video()

    # display.display(display.HTML(video))
    # plt.close()
    ani.save(cwd + '/animSLE.mp4', writer='ffmpeg')
