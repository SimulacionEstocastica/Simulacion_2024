import numpy as np
import matplotlib.pyplot as plt
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

        bridge = brownian_path[-1]

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


    def plot_graph(self):
        try:
            pos = nx.get_node_attributes(self.graph, 'pos')

            node_colors = [
                'green' if node == self.initial_node else 
                'red' if node in self.frontera else 
                'blue' 
                for node in self.graph.nodes
            ]

            plt.figure(figsize=(8, 8))
            nx.draw(self.graph, pos, with_labels=False, node_size=50, node_color=node_colors)
            plt.show()

        except Exception as e:
            print(f"Error al intentar dibujar el grafo: {e}")

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

        while graph.node not in graph.frontera:
            lengths_dict = graph.length_adjacent_edges(graph.node)
            time, brownian_path = graph.simulate_brownian(min(lengths_dict.values()))
            time, brownian_path = graph.convert_brownian_to_graph(time, brownian_path, lengths_dict)


        phis[n] = graph.phi()

        graph.node = node
        graph.hitting_time = 0

    return np.mean(phis)

def PlotGraph3d(graph, node_values, **kwargs):
        
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
        
    x_vals = [node[0] for node in graph.graph.nodes]
    y_vals = [node[1] for node in graph.graph.nodes]
        
    z_vals = [node_values.get(node, 0) for node in graph.graph.nodes] 
        
    ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=100, **kwargs)
        
    for edge in graph.graph.edges:
        x_edge = [edge[0][0], edge[1][0]]
        y_edge = [edge[0][1], edge[1][1]]
        z_edge = [node_values.get(edge[0], 0), node_values.get(edge[1], 0)]
        ax.plot(x_edge, y_edge, z_edge, color='gray', lw=1)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Phi')
        
    plt.show()

def FuncionArmonica(N, delta_t):

    dict_phis = {}

    brownian_graph = BrownianOverGraph(initial_node = (5, 5), delta_t = delta_t)

    brownian_graph.graph = GridGraph(10,10)
    brownian_graph.frontera = GridFrontier(10,10)

    brownian_graph.plot_graph()

    for node in brownian_graph.graph.nodes:
            print(node)
            dict_phis[node] = CalcularEsperanza(N = N, graph = brownian_graph, node=node)
    
    print("Laplaciano Discreto: ", ComprobarArmonicidad(brownian_graph.graph, dict_phis, brownian_graph.frontera))

    PlotGraph3d(brownian_graph, dict_phis)

def ComprobarArmonicidad(graph, phis, frontera):
    def LaplacianoDiscreto(graph, node, phis):
        return sum(phis[node] - phis[w] for w in graph.neighbors(node))
    vect = []
    for node in graph.nodes:
        if node not in frontera:
            vect.append(LaplacianoDiscreto(graph, node, phis))
    
    return np.linalg.norm(vect, 2)
    
    return np.linalg.norm(vect, 2)

start_time = time.time()
FuncionArmonica(N = 500, delta_t = 0.01)
end_time = time.time()

execution_time = end_time - start_time

print(f"Tiempo de ejecución: {execution_time:.5f} segundos")

###########################################################################################################