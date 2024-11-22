'''   Funciones para la creación de Grafos y determinación de Fronteras   '''

# Importar Librerías
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
from scipy.spatial import Delaunay


######################## Grid ##########################

def GridGraph(N1, N2):
    G = nx.grid_graph(dim=(N1, N2))
    
    for node in G.nodes:
        G.nodes[node]['pos'] = node 

    return G

def GridFrontier(N1,N2):

    frontera = [(0, j) for j in range(N1)] + [(N2-1, j) for j in range(N1)]
    
    frontera += [(i, 0) for i in range(1, N2-1)] + [(i, N1-1) for i in range(1, N2-1)]
    
    return frontera

########################################################




###################### Sierpinski ######################

def SierpinskiGraph(depth, G=None, p1=(0, 0), p2=(1, 0), p3=(0.5, 1)):
    if G is None:
        G = nx.Graph()
    
    if depth == 0:
        G.add_edges_from([(p1, p2), (p2, p3), (p3, p1)])

    else:
        mid12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        mid23 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        mid31 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
        
        SierpinskiGraph(depth - 1, G, p1, mid12, mid31)  
        SierpinskiGraph(depth - 1, G, mid12, p2, mid23) 
        SierpinskiGraph(depth - 1, G, mid31, mid23, p3) 
    
    for node in G.nodes:
        G.nodes[node]['pos'] = node
        
    return G

def SierpinskiFrontier(depth):

    frontera = [(j, 0.0) for j in np.linspace(0.0, 1.0, num = 2**(depth) + 1)] 

    frontera += [(j/2, j) for j in np.linspace(0.0, 1.0, num=2**depth + 1)]
    
    frontera += [(0.5 + j/2, 1.0 - j) for j in np.linspace(0.0, 1.0, num=2**depth + 1)]

    return frontera

SierpinskiFrontier2 = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

########################################################




################### Circle - Delaunay ##################

def DelaunayCircleGraph(n_outer, n_inner):

    angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer_points = np.column_stack((np.cos(angles), np.sin(angles)))

    radii = np.sqrt(np.random.rand(n_inner))  
    inner_angles = 2 * np.pi * np.random.rand(n_inner)
    inner_points = np.column_stack((radii * np.cos(inner_angles), radii * np.sin(inner_angles)))

    points = np.vstack((outer_points, inner_points))

    tri = Delaunay(points)

    G = nx.Graph()

    for point in points:
        G.add_node(tuple(point), pos=point)  # Usar el par (x, y) como identificador del nodo

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(tuple(points[simplex[i]]), tuple(points[simplex[j]]))
                
    return G

def DelaunayCircleFrontier(n_outer):

    angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)

    frontera = [(np.cos(angle), np.sin(angle)) for angle in angles]

    return frontera
        
########################################################




################### Circle - Rings #####################

def RingsCircleGraph(anillos, nodos_por_anillo):
    G = nx.Graph()

    for i in range(anillos):
        theta = np.linspace(0, 2 * np.pi, nodos_por_anillo, endpoint=False)
        radio = i + 1
        
        for j in range(nodos_por_anillo):
            x, y = radio * np.cos(theta[j]), radio * np.sin(theta[j])
            G.add_node((x, y), pos=(x, y), label=(round(x, 2), round(y, 2)))
            G.add_edge((x, y), (radio * np.cos(theta[(j + 1) % nodos_por_anillo]), radio * np.sin(theta[(j + 1) % nodos_por_anillo])))

            if i < anillos - 1:
                G.add_edge((x, y), ((radio + 1) * np.cos(theta[j]), (radio + 1) * np.sin(theta[j])))

    return G

def RingsCircleFrontier(nodos_por_anillo, levels):

    frontera = []

    for l in levels:
        theta = np.linspace(0, 2 * np.pi, nodos_por_anillo, endpoint=False)

        for j in range(nodos_por_anillo):
            x, y =  l * np.cos(theta[j]), l * np.sin(theta[j])
            frontera.append((x,y))

    return frontera

########################################################




############## Circle - Rings - Triangulation ###########

def TriangulationsRingsCircleGraph(n_outer, n_inner_layers):
    G = nx.Graph()

    outer_nodes = []
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        x, y = np.cos(angle), np.sin(angle)
        G.add_node((x, y), pos=(x, y))  
        outer_nodes.append((x, y))


    for i in range(n_outer):
        G.add_edge(outer_nodes[i], outer_nodes[(i + 1) % n_outer])

    inner_nodes = []
    layer_nodes = outer_nodes

    for layer in range(n_inner_layers):
        next_layer_nodes = []
        num_nodes = len(layer_nodes)
        
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            radius = (n_inner_layers - layer) / (n_inner_layers + 1)
            x, y = radius * np.cos(angle), radius * np.sin(angle)

            new_node = (x, y) 
            G.add_node(new_node, pos=(x, y))

            next_layer_nodes.append(new_node)

            G.add_edge(new_node, layer_nodes[i])
            G.add_edge(new_node, layer_nodes[(i + 1) % num_nodes])

            if i > 0:
                G.add_edge(new_node, next_layer_nodes[i - 1])

        G.add_edge(next_layer_nodes[0], next_layer_nodes[-1])

        layer_nodes = next_layer_nodes
        inner_nodes.extend(next_layer_nodes)

    return G

def TriangulationsRingsCircleFrontier(G, n_outer):

    frontera = []

    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        x, y = np.cos(angle), np.sin(angle)
        G.add_node((x, y), pos=(x, y))  
        frontera.append((x, y))
    
    return frontera

########################################################




###################### Hexagonal ######################

def HexaGraph(hex_size=0.2):
    G = nx.Graph()

    points = []
    for i in range(-int(1 / 0.05), int(1 / 0.05) + 1):
        for j in range(-int(1 / 0.05), int(1 / 0.05) + 1):
        
            x = i * 0.05 * np.sqrt(3) / 2
            y = j * 0.05 + (i % 2) * 0.05 / 2
            if x**2 + y**2 <= 1:
                points.append((x, y))

    for point in points:
        G.add_node(point, pos=point)

    for i, (x1, y1) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            if i < j and np.isclose(np.sqrt((x2 - x1)**2 + (y2 - y1)**2), 0.5):
                G.add_edge((x1, y1), (x2, y2))

    return G

def HexaFrontier(hex_size=0.2):
    G = HexaGraph(hex_size)  
    frontera = []

    max_dist = max(np.sqrt(x**2 + y**2) for x, y in G.nodes)

    for node, data in G.nodes(data=True):
        x, y = node
        dist = np.sqrt(x**2 + y**2)

        if np.isclose(dist, max_dist, atol=hex_size-0.01): 
            frontera.append(node)

    return frontera

########################################################




################### Hexagonal Laticce ###################

def HexaLatticeGraph(m, n):
    # Crear el grafo hexagonal
    G = nx.hexagonal_lattice_graph(m, n)
    
    # Asignar coordenadas (x, y) a cada nodo
    pos = nx.get_node_attributes(G, 'pos')
    G = nx.relabel_nodes(G, {node: pos[node] for node in G.nodes()})
    
    return G

def HexaLatticeFrontier(m, n):
    if n % 2 == 0: 
        frontera = [(0.0, 0.8660254037844386), 
                    (0.0, 1.7320508075688772*(m)- 0.8660254037844386), 
                    ((n/2)*2 + (n/2 + 0.5) ,1.7320508075688772), 
                    ((n/2)*2 + (n/2 + 0.5), 1.7320508075688772*(m))]
    else:
        frontera = [(0.0, 0.8660254037844386), 
                    (0.0, 0.8660254037844386 + 1.7320508075688772*(m-1)), 
                    ((n/2 + 0.5)*2 + (n/2 - 0.5) ,0.8660254037844386), 
                    ((n/2 + 0.5)*2 + (n/2 - 0.5), 0.8660254037844386 + 1.7320508075688772*(m-1))]


    return frontera

########################################################




##################### Koch Border #####################

def KochBorderGraph(depth):
    def add_edges(G, p1, p2, depth):
        if depth == 0:
            G.add_edge(p1, p2)
            G.nodes[p1]['pos'] = p1 
            G.nodes[p2]['pos'] = p2 
        else:
            x1, y1 = p1
            x2, y2 = p2

            pA = ((2*x1 + x2) / 3, (2*y1 + y2) / 3)
            pB = ((x1 + 2*x2) / 3, (y1 + 2*y2) / 3)
            
            angle = math.atan2(y2 - y1, x2 - x1) - math.pi / 3
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 3
            pC = (pA[0] + dist * math.cos(angle), pA[1] + dist * math.sin(angle))
            
            add_edges(G, p1, pA, depth - 1)
            add_edges(G, pA, pC, depth - 1)
            add_edges(G, pC, pB, depth - 1)
            add_edges(G, pB, p2, depth - 1)

    G = nx.Graph()
    
    p1 = (0, 0)
    p2 = (1, 0)
    p3 = (0.5, math.sqrt(3) / 2)
    
    add_edges(G, p1, p2, depth)
    add_edges(G, p2, p3, depth)
    add_edges(G, p3, p1, depth)
    
    return G

def KochBorderFrontier():
    frontera = [(0, 0), 
                (0.0, 0.5773502691896257), 
                (0.5, 0.8660254037844386),
                (1.0, 0.5773502691896257),
                (1, 0),
                (0.5, -0.28867513459481287)]
    return frontera


########################################################




##################### Koch Ring #####################

def KochRingsGraph(num_rings, initial_radius, ring_depth, num_sides=6):

    def add_koch_edges(G, p1, p2, depth):
        if depth == 0:
            G.add_edge(p1, p2)
            G.nodes[p1]['pos'] = p1  
            G.nodes[p2]['pos'] = p2  
            
            return [p1, p2]
        else:
            x1, y1 = p1
            x2, y2 = p2
            pA = ((2 * x1 + x2) / 3, (2 * y1 + y2) / 3)
            pB = ((x1 + 2 * x2) / 3, (y1 + 2 * y2) / 3)
            
            angle = math.atan2(y2 - y1, x2 - x1) - math.pi / 3
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 3
            pC = (pA[0] + dist * math.cos(angle), pA[1] + dist * math.sin(angle))
            
            points = []
            points += add_koch_edges(G, p1, pA, depth - 1)
            points += add_koch_edges(G, pA, pC, depth - 1)
            points += add_koch_edges(G, pC, pB, depth - 1)
            points += add_koch_edges(G, pB, p2, depth - 1)
            
            return points

    def generate_koch_ring(G, center, radius, num_sides, depth):
        points = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        
        ring_points = []
        for i in range(num_sides):
            p1 = points[i]
            p2 = points[(i + 1) % num_sides]
            ring_points += add_koch_edges(G, p1, p2, depth)
        
        return ring_points

    G = nx.Graph()
    center = (0, 0)
    
    last_ring_points = generate_koch_ring(G, center, initial_radius, num_sides, ring_depth)
    
    for ring in range(1, num_rings):

        radius = initial_radius * (ring + 1)
        
        current_ring_points = generate_koch_ring(G, center, radius, num_sides, ring_depth)
        
        num_points = min(len(last_ring_points), len(current_ring_points))
        for i in range(num_points):
            G.add_edge(last_ring_points[i], current_ring_points[i])

        last_ring_points = current_ring_points
    
    return G

def KochRingsFrontier(G, initial_radius):

    center = (0, 0)
    frontera = []
    
    for node in G.nodes:
        x, y = node
        distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        
        if math.isclose(distance, initial_radius, rel_tol=0.3):
            frontera.append(node)
        
    max_dist = max(np.sqrt(x**2 + y**2) for x, y in G.nodes)

    for node, data in G.nodes(data=True):
        x, y = node
        dist = np.sqrt(x**2 + y**2)

        if np.isclose(dist, max_dist, atol=1.1): 
            frontera.append(node)
    
    return frontera

def KochRingsFrontier2(G, initial_radius):

    frontera = []
        
    max_dist = max(np.sqrt(x**2 + y**2) for x, y in G.nodes)

    for node, data in G.nodes(data=True):
        x, y = node
        dist = np.sqrt(x**2 + y**2)

        if np.isclose(dist, max_dist, atol=0.03): 
            frontera.append(node)
    
    return frontera

########################################################




##################### Cross Graph #####################

def CrossGraph(depth, branching_factor):
    def add_hyperbolic_layer(G, node, depth, max_depth, branching_factor):
        if depth >= max_depth:
            return
        
        for i in range(branching_factor):
            angle = 2 * np.pi * i / branching_factor
            radius = (depth + 1) / max_depth
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            new_node = (round(x, 2), round(y, 2))
            G.add_node(new_node, pos=(x, y))
            G.add_edge(node, new_node)
            
            add_hyperbolic_layer(G, new_node, depth + 1, max_depth, branching_factor)

    G = nx.Graph()
    root_pos = (0, 0)
    G.add_node(root_pos, pos=root_pos)
    add_hyperbolic_layer(G, root_pos, 0, depth, branching_factor)
    return G

def CrossFrontier(G):
    frontera = []
    max_dist = max(np.sqrt(x**2 + y**2) for x, y in G.nodes)

    for node, data in G.nodes(data=True):
        x, y = node
        dist = np.sqrt(x**2 + y**2)

        if np.isclose(dist, max_dist, atol=0.1): 
            frontera.append(node)
            
    return frontera

########################################################




###################### Spin Graph ######################

def SpinGraph(num_anillos, puntos_por_anillo):
    G = nx.Graph()
    
    for i in range(num_anillos):
        radio = (i + 1) / num_anillos 
        for j in range(puntos_por_anillo):
            angulo = 2 * np.pi * j / puntos_por_anillo  
            x = round(radio * np.cos(angulo), 2) 
            y = round(radio * np.sin(angulo), 2) 
            pos = (x, y)  
            G.add_node(pos, pos=(x, y)) 
    

    for i in range(num_anillos - 1):
        for j in range(puntos_por_anillo):

            x1 = round((i + 1) / num_anillos * np.cos(2 * np.pi * j / puntos_por_anillo), 2)
            y1 = round((i + 1) / num_anillos * np.sin(2 * np.pi * j / puntos_por_anillo), 2)
            node1 = (x1, y1)

            x2 = round((i + 2) / num_anillos * np.cos(2 * np.pi * j / puntos_por_anillo), 2)
            y2 = round((i + 2) / num_anillos * np.sin(2 * np.pi * j / puntos_por_anillo), 2)
            node2 = (x2, y2)

            x3 = round((i + 2) / num_anillos * np.cos(2 * np.pi * (j + 1) / puntos_por_anillo), 2)
            y3 = round((i + 2) / num_anillos * np.sin(2 * np.pi * (j + 1) / puntos_por_anillo), 2)
            node3 = (x3, y3)

            G.add_edge(node1, node2)
            G.add_edge(node1, node3)

    return G

def SpinFrontier(G): 
    frontera = []
    max_dist = max(np.sqrt(x**2 + y**2) for x, y in G.nodes)

    for node, data in G.nodes(data=True):
        x, y = node
        dist = np.sqrt(x**2 + y**2)

        if np.isclose(dist, max_dist, atol=0.05): 
            frontera.append(node)
            
    return frontera

########################################################
