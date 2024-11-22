
import numpy as np

from Ant import Ant
from Graphicator import Graphicator
from matplotlib import pyplot as plt
from Map import Map

#a = Map(np.array([[0, 1, 10,1], [1, 0, 1,2], [10, 1, 0, 1], [1,2,1,0]]), 0.3, 4, 4, 0.8, 0.7, 0.1)
#pheromones = a.pheromones
#ant = Ant(1, a, 0)
#ant2 = Ant(2, a, 2)
#a.set_ants([ant, ant2])
#a.iterator()
#
#b = Graphicator()
#b.city_creator(np.array([(1,0),(0,0),(0,1),(1,1)]))
#b.plot_points()
#b.plot_all([ant2])

q_0 = 0.8 # Probabilidad de que las hormigas escojan la siguiente ciudad maximizando τil(t) · [ηil]^β
alpha = 1 # Influencia de las feromonas sobre la decisión de las hormigas (si es grande, las hormigas prefieren caminos con mas feromonas)
beta = 3  # Influencia de las distancias sobre la decisión de las hormigas (si es grande, se tiende a elegir ciudades cercanas)
rho = 0.7 #antes 0.5 # Tasa de evaporación de las feromonas
chi = 0.2 #Tasa de perdida de interés sobre arcos recien usados (si es grande, las hormigas tienden a explorar mas caminos)
t_o = 0.05 #antes 0.2 # Constante tal que chi · t_o se le suma a las feromonas de los arcos recien usados
num_of_ants = 2 #antes 5
num_of_cities = 4
positions = np.array(range(num_of_ants))
graph = Graphicator()
graph.city_creator(np.array([[1,1],
                             [2.5,4],
                             [4,1],
                             [2.5,2]]))
map = Map(graph.mat_adj, q_0, alpha, beta, rho, chi, t_o)
ants_tot = []
#pheromones_tot = []
best_ant_dist = []
ants_tot_ant = []

def q0():
    global q_0
    q_0 = float(input("Nuevo valor de q0: "))

def alfa():
    global alpha
    alpha = float(input("Nuevo valor de alpha: "))

def betaa():
    global beta
    beta = float(input("Nuevo valor de beta: "))

def ro():
    global rho
    rho = float(input("Nuevo valor de rho: "))

def chii():
    global chi
    chi = float(input("Nuevo valor de chi: "))

def t0():
    global t_o
    t_o = float(input("Nuevo valor de t0: "))
def ants():
    global num_of_ants
    num_of_ants = int(input("Nuevo valor de num_of_ants: "))

def new_parameter(parameter):
    if parameter == 'q_0':
        q0()
        return 1
    elif parameter == 'alpha':
        alfa()
        return 1
    elif parameter == 'beta':
        betaa()
        return 1
    elif parameter == 'rho':
        ro()
        return 1
    elif parameter == 'chi':
        chii()
        return 1
    elif parameter == 't_o':
        t0()
        return 1
    elif parameter == 'num_of_ants':
        ants()
        return 1
    else:
        return 0

def func_1(n_ants, n_cities):
    global num_of_ants, num_of_cities, positions
    num_of_ants = n_ants
    num_of_cities = n_cities
    positions = np.array(range(num_of_ants))

def func_2(l):
    global graph, num_of_cities
    graph.city_creator_random(num_of_cities,l)

def func_3():
    global map, q_0, alpha, beta, rho, chi, t_o
    map = Map(graph.mat_adj, q_0, alpha, beta, rho, chi, t_o)

def clean():
    global ants_tot,best_ant_dist, ants_tot_ant#, pheromones_tot
    ants_tot = []
    #pheromones_tot = []
    best_ant_dist = []
    ants_tot_ant = []

def iterator(n, update=False):
    global positions, map, num_of_ants, ants_tot, best_ant_dist# pheromones_tot
    for i in range(n):
        if update:
            positions = np.random.randint(0, num_of_cities, size=num_of_ants)
        ants = [Ant(i,map,positions[i]) for i in range(num_of_ants)]
        map.set_ants(ants)
        map.iterator()
        ants_tot.append(map.best_ant.tour)
        ants_tot_ant.append(map.best_ant)
        best_ant_dist.append(map.best_ant.current_distance)
        #pheromones_tot.append(map.get_max_pheromone_tour())


def plot():
    global graph, map, ants_tot, best_ant_dist, ants_tot_ant
    print("Puntos")
    graph.plot_points()
    print("Tour stocastic")
    index = -1
    best = 100000000
    for i in range(len(ants_tot)):
        if best_ant_dist[i] < best:
            index = i
            best = best_ant_dist[i]

    graph.plot_tour(ants_tot_ant[index],edges=False)
    print("Best tours in rounds")
    graph.plot_tours(ants_tot,edges=False)
    r = input("¿Desea comparar con la solución determinista?")
    if r:
        #graph.plot_pheromone_max(map.get_max_pheromone_tour(),edges=False)
        print("Tour pulp")
        pulp_best_tour, pulp_best_tour_distance = graph.pulp_solution()
        print("Distance Tours vs Distance Tour pulp")
        graph.plot_distances(best_ant_dist, pulp_best_tour_distance)
    print("Pheromones in rounds")
    graph.plot_final_pheromones(map.pheromones.history, edges=False)
    #graph.plot_max_pheromone_tours(pheromones_tot,edges=False)

def setup():
    r1 = input("¿Desea trabajar con el grafo por defecto? (si/no) \n Respuesta: ")
    if r1 == "no":
        n_cities= int(input("Ingrese el número de ciudades.\n Respuesta: "))
        n_ants= int(input("Ingrese el número de hormigas.\n Respuesta: "))
        func_1(n_ants, n_cities)
        l = float(input("Ingresar el tamaño del plano.\n Respuesta: "))
        func_2(l)


def run():
    global positions, num_of_ants, num_of_cities
    r1 = input("¿Desea trabajar con los mismos parámetros? (si/no) \n Respuesta: ")
    if r1 == "no":
        r = 1
        while r != 0 :
            parameter = input("Parámetro a cambiar: (escriba un número si no cambiará otro) \n Respuesta: ")
            r = new_parameter(parameter)
    func_3()
    r0 = input("¿Desea actualizar aleatoriamente la posicion de las hormigas en cada iteración? \n Respuesta: ")
    if r0 == "no":
        n = int(input("¿Cuántas iteraciones? \n Respuesta: "))
        iterator(n)
        plot()
        clean()
        func_3()
    else:
        n = int(input("¿Cuántas iteraciones? \n Respuesta: "))
        iterator(n, True)
        plot()
        clean()
        func_3()

setup()
s = True
while s:
    run()
    clean()
    res = input("¿Desea seguir trabajando con este grafo? (si/no) \n Respuesta: ")
    if res == "no":
        s = False