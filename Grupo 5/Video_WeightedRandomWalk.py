import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import time
from graphs import *


################################## DEFINICIÓN DE LA CLASE #############################################


class SimetricRandomWalkOverGraph():
    def __init__(self, initial_node):
        self.graph = None
        self.initial_node = initial_node
        self.node = initial_node 
        self.frontera = None

    def simulate_random_walk(self, node):
        neighbors = [v for v in self.graph.neighbors(node)]
        index_choice = np.random.choice([i for i in range(len(neighbors))])
        self.node = neighbors[index_choice]
        
        
    def phi(self):
        return np.linalg.norm((self.initial_node[0] - self.node[0], self.initial_node[1] - self.node[1]), 2)


    def plot_graph(self, **kwargs):
        try:
            pos = nx.get_node_attributes(self.graph, 'pos')

            node_colors = ['red' if node in self.frontera  else 'blue' for node in self.graph.nodes]

            plt.figure(figsize=(8, 8))
            nx.draw(self.graph, pos, with_labels=False, node_size=50, node_color=node_colors)
            plt.show()

        except Exception as e:
            print(f"Error al intentar dibujar el grafo: {e}")

##################################################################################################




################################### Código Principal #############################################


def CalcularEsperanza(N, graph, node):

    phis = np.zeros(N)

    for n in range(N):

        graph.node = node

        while graph.node not in graph.frontera:
            rw_path = graph.simulate_random_walk(graph.node)
            graph.hitting_time += 1

        phis[n] = graph.phi()

        graph.node = node
        graph.hitting_time = 0

    return np.mean(phis)

def PlotGraph3dAnimated(graph, node_values_func, num_frames, interval=200, **kwargs):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    nodes_list = list(graph.graph.nodes)

    x_vals = [node[0] for node in nodes_list]
    y_vals = [node[1] for node in nodes_list]

    z_vals = [0 for _ in nodes_list]

    scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=100, **kwargs)
    edge_lines = []

    for edge in graph.graph.edges:
        x_edge = [edge[0][0], edge[1][0]]
        y_edge = [edge[0][1], edge[1][1]]
        z_edge = [0, 0]  
        edge_line, = ax.plot(x_edge, y_edge, z_edge, color='gray', lw=2)
        edge_lines.append(edge_line)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Phi')

    def update(frame):
        nonlocal scatter, edge_lines

        scatter.remove()

        for edge_line in edge_lines:
            edge_line.remove()

        if frame == 0:
            z_vals = [0 for _ in nodes_list]
        else:
            new_node_values = node_values_func(frame - 1)
            z_vals = [new_node_values.get(node, 0) for node in nodes_list]


        scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=100, **kwargs)

        edge_lines = []
        for edge in graph.graph.edges:
            x_edge = [edge[0][0], edge[1][0]]
            y_edge = [edge[0][1], edge[1][1]]
            z_edge = [z_vals[nodes_list.index(edge[0])], z_vals[nodes_list.index(edge[1])]]
            edge_line, = ax.plot(x_edge, y_edge, z_edge, color='gray', lw=2)
            edge_lines.append(edge_line)

        ax.set_title(f"Iteración {frame}")
        return scatter, edge_lines

    # Crear animación
    anim = FuncAnimation(fig, update, frames=num_frames + 1, interval=interval, blit=False)
    
    return anim


def FuncionArmonicaAnimada(N):
    N1 = 30
    N2 = 30

    rw_graph = SimetricRandomWalkOverGraph(initial_node=(N1//2,N2//2))

    rw_graph.graph = GridGraph(N1,N2)
    rw_graph.frontera = GridFrontier(N1,N2)

    dict_phis = {node: [] for node in rw_graph.graph.nodes}

    def update_phi(frame):
        for node in rw_graph.graph.nodes:
            print(node)
            dict_phis[node].append(CalcularEsperanza(N=1, graph=rw_graph, node=node))
        return {node: np.mean(dict_phis[node]) for node in rw_graph.graph.nodes}

    anim = PlotGraph3dAnimated(rw_graph, update_phi, num_frames=N, interval=200)
    return anim


anim = FuncionArmonicaAnimada(N = 100)

anim.save("Grid(30x30)_weightedrandomwalk.mp4", fps = 20, dpi=300, extra_args=['-vcodec', 'libx264'])

###########################################################################################################