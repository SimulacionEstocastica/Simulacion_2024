import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import time
from graphs import *


################################## DEFINICIÓN DE LA CLASE #############################################


class BrownianOverGraph():
    def __init__(self, initial_node, delta_t):
        self.graph = None
        self.initial_node = initial_node
        self.node = initial_node 
        self.frontera = None
        self.hitting_time = 0
        self.delta_t = delta_t

    def length_adjacent_edges(self, v):
        lengths_dict = {}
        for node in self.graph.neighbors(v):
            lengths_dict[node] = np.linalg.norm((v[0] - node[0], v[1] - node[1]), 2)
        return lengths_dict

    def simulate_brownian(self, min_length):
        time = [0]
        brownian_path = [0]
        sigma = np.sqrt(self.delta_t)
        while abs(brownian_path[-1]) < min_length:
            time.append(time[-1] + self.delta_t)
            brownian_path.append(brownian_path[-1] + np.random.normal(0, sigma))
        return  time, brownian_path

    def convert_brownian_to_graph(self, time, brownian_path, lengths_dict):

        #Determinar el Bridge
        bridge = brownian_path[-1]

        #Asignar aleatoriamente el Bridge a uno de los lenghts
        bridge_asigned_index = np.random.choice([i for i in range(len(lengths_dict))])
        node_bridge_asigned = list(lengths_dict)[bridge_asigned_index]
        length_assigned = lengths_dict.get(node_bridge_asigned)

        if length_assigned <= abs(bridge):
            self.node = node_bridge_asigned
            self.hitting_time += time[-1]
        else:
            sigma = np.sqrt(self.delta_t)
            while True:
                time.append(time[-1] + self.delta_t)
                brownian_path.append(brownian_path[-1] + np.random.normal(0, sigma))
                if abs(brownian_path[-1]) >= length_assigned: 
                    self.node = node_bridge_asigned
                    break
                elif np.sign(brownian_path[-2]*brownian_path[-1]) == -1: break
        
        return time, brownian_path

    def phi(self):
        return np.linalg.norm((self.initial_node[0] - self.node[0], self.initial_node[1] - self.node[1]), 2)


    def plot_graph(self, **kwargs):
        pos = {node: node for node in self.graph.nodes()}
        plt.figure(figsize=(8, 8))
        nx.draw(self.graph, pos, **kwargs)
        plt.show()

    def plot_brownian(self, time, brownian_path):
        plt.figure(figsize=(8, 8))
        plt.plot(time, brownian_path, color='blue', linewidth=2, label='Brownian Path')
        plt.title('Brownian Motion in R')
        plt.xlabel('t')
        plt.ylabel('B(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

##################################################################################################




################################### Código Principal #############################################


def CalcularEsperanza(N, graph, node):

    phis = np.zeros(N)

    for n in range(N):

        graph.node = node

        # Realización del browniano sobre el grafo hasta llegar a la frontera
        while graph.node not in graph.frontera:
            lengths_dict = graph.length_adjacent_edges(graph.node)
            time, brownian_path = graph.simulate_brownian(min(lengths_dict.values()))
            time, brownian_path = graph.convert_brownian_to_graph(time, brownian_path, lengths_dict)


        phis[n] = graph.phi()

        graph.node = node
        graph.hitting_time = 0

    return np.mean(phis)

def PlotGraph3dAnimated(graph, node_values_func, num_frames, interval=200, **kwargs):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convertir los nodos a una lista para un acceso más sencillo
    nodes_list = list(graph.graph.nodes)

    # Configurar límites iniciales
    x_vals = [node[0] for node in nodes_list]
    y_vals = [node[1] for node in nodes_list]

    # Iniciar valores z en 0 para el primer frame
    z_vals = [0 for _ in nodes_list]

    # Crear scatter inicial
    scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=100, **kwargs)
    edge_lines = []

    # Graficar las aristas iniciales
    for edge in graph.graph.edges:
        x_edge = [edge[0][0], edge[1][0]]
        y_edge = [edge[0][1], edge[1][1]]
        z_edge = [0, 0]  # Inicialmente las aristas estarán en z=0
        edge_line, = ax.plot(x_edge, y_edge, z_edge, color='gray', lw=2)
        edge_lines.append(edge_line)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Phi')

    # Función de actualización
    def update(frame):
        nonlocal scatter, edge_lines

        # Eliminar scatter anterior
        scatter.remove()

        # Eliminar líneas de aristas anteriores
        for edge_line in edge_lines:
            edge_line.remove()

        # Para la primera iteración, mantener z_vals en 0
        if frame == 0:
            z_vals = [0 for _ in nodes_list]
        else:
            # Actualizar valores de nodos para frames posteriores
            new_node_values = node_values_func(frame - 1)
            z_vals = [new_node_values.get(node, 0) for node in nodes_list]

        # Crear nuevo scatter
        scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=100, **kwargs)

        # Actualizar las aristas
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


def FuncionArmonicaAnimada(N, delta_t):
    N1 = 30
    N2 = 30
    brownian_graph = BrownianOverGraph(initial_node=(N1//2,N2//2), delta_t=delta_t)
    brownian_graph.graph = GridGraph(N1,N2)
    brownian_graph.frontera = GridFrontier(N1,N2)
    dict_phis = {node: [] for node in brownian_graph.graph.nodes}

    # Función para calcular los valores de los nodos en cada frame
    def update_phi(frame):
        for node in brownian_graph.graph.nodes:
            print(node)
            dict_phis[node].append(CalcularEsperanza(N=1, graph=brownian_graph, node=node))
        return {node: np.mean(dict_phis[node]) for node in brownian_graph.graph.nodes}

    # Generar la animación
    anim = PlotGraph3dAnimated(brownian_graph, update_phi, num_frames=N, interval=200)
    return anim


anim = FuncionArmonicaAnimada(N = 100, delta_t = 0.01)

anim.save("Grid(30x30)_browniano.mp4", fps = 20, dpi=300, extra_args=['-vcodec', 'libx264'])

###########################################################################################################