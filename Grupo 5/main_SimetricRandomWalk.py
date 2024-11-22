import numpy as np
import matplotlib.pyplot as plt
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
        #return np.cos(5*np.arctan(self.node[1]/self.node[0]))
        


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

##################################################################################################




################################### Código Principal #############################################


def CalcularEsperanza(N, graph, node):

    phis = np.zeros(N)

    for n in range(N):

        graph.node = node

        while graph.node not in graph.frontera:
            rw_path = graph.simulate_random_walk(graph.node)

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
        
    ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=50, **kwargs)
        
    for edge in graph.graph.edges:
        x_edge = [edge[0][0], edge[1][0]]
        y_edge = [edge[0][1], edge[1][1]]
        z_edge = [node_values.get(edge[0], 0), node_values.get(edge[1], 0)]
        ax.plot(x_edge, y_edge, z_edge, color='gray', lw=1)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Phi')
        
    plt.show()

def FuncionArmonica(N):

    dict_phis = {}

    rw_graph = SimetricRandomWalkOverGraph(initial_node = ((10,10)))


    rw_graph.graph = GridGraph(20,20)
    rw_graph.frontera = GridFrontier(20,20)

    rw_graph.plot_graph()

    for node in rw_graph.graph.nodes:
        print(node)
        dict_phis[node] = CalcularEsperanza(N = N, graph = rw_graph, node=node)

    print("Laplaciano Discreto: ", ComprobarArmonicidad(rw_graph.graph, dict_phis, rw_graph.frontera))

    PlotGraph3d(rw_graph, dict_phis)

def ComprobarArmonicidad(graph, phis, frontera):
    def LaplacianoDiscreto(graph, node, phis):
        return sum(phis[node] - phis[w] for w in graph.neighbors(node))
    vect = []
    for node in graph.nodes:
        if node not in frontera:
            vect.append(LaplacianoDiscreto(graph, node, phis))
    
    return np.linalg.norm(vect, 2)

start_time = time.time()
FuncionArmonica(N = 10000)
end_time = time.time()

execution_time = end_time - start_time

print(f"Tiempo de ejecución: {execution_time:.5f} segundos")

###########################################################################################################