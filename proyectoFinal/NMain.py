import numpy as np
from Graphicator import Graphicator
from matplotlib import pyplot as plt
from NMap import Map
from NAnt import AntA
from NAnt import AntB


q_0 = 0.5               # Probabilidad de que las hormigas escojan la siguiente ciudad maximizando τil(t) · [ηil]^β
alpha = 0.5             # Influencia de las feromonas sobre la decisión de las hormigas (si es grande, las hormigas prefieren caminos con mas feromonas)
beta = 2                # Influencia de las distancias sobre la decisión de las hormigas (si es grande, se tiende a elegir ciudades cercanas)
rho = 0.7               #antes 0.5 # Tasa de evaporación de las feromonas
chi = 0.2               #Tasa de perdida de interés sobre arcos recien usados (si es grande, las hormigas tienden a explorar mas caminos)
t_o = 0.05               #antes 0.2 # Constante tal que chi · t_o se le suma a las feromonas de los arcos recien usados
gamma = 1                  ##############  #### ## ## ## ## ## ## ## ## ########### 333 333 3333 33
epsilon = 1e-5             ##############  #### ## ## ## ## ## ## ## ## ########### 333 333 3333 33
phi = 0.5                  ##############  #### ## ## ## ## ## ## ## ## ########### 333 333 3333 33
num_of_ants_x_colony = 10 ##############  #### ## ## ## ## ## ## ## ## ########### 333 333 3333 33
num_of_cities = 6         ##############  #### ## ## ## ## ## ## ## ## ########### 333 333 3333 33
positionsA = np.random.randint(0, 3, size=num_of_ants_x_colony)
positionsB = np.random.randint(5, 6, size=num_of_ants_x_colony)
graph = Graphicator()
graph.city_creator(np.array([[1,1],
                                [1,4],
                                [4,1.5],
                                [4,3.5],
                                [7,4],
                                [7,1]]))
objectiveA1 = [0,1,2]
objectiveB1 = [3,4,5]
objectiveA2 = [0,1,2,3]
objectiveB2 = [2,3,4,5]
objectiveA3 = [1,3,4,0]
objectiveB3 = [1,3,4,5]
objectiveA4 = [1,3,4]
objectiveB4 = [1,3,4]

objectiveA = objectiveA1
objectiveB = objectiveB1

map = Map(graph.mat_adj, q_0, alpha, beta, gamma, epsilon, rho, chi, t_o, phi)
map.change_objectiveA(objectiveA1)
map.change_objectiveB(objectiveB1)

ants_totA = []
ants_totB = []
best_ant_distA = []
best_ant_distB = []
best_ant_dist_tourA = []
best_ant_dist_tourB = []

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
    global num_of_ants_x_colony
    num_of_ants_x_colony = int(input("Nuevo valor de num_of_ants: "))

def gama():
    global t_o
    t_o = float(input("Nuevo valor de gamma: "))

def eps():
    global t_o
    t_o = float(input("Nuevo valor de epsilon: "))

def fi():
    global t_o
    t_o = float(input("Nuevo valor de phi: "))

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
    elif parameter == 'gamma':
        gama()
        return 1
    elif parameter == 'epsilon':
        eps()
        return 1
    elif parameter == 'phi':
        fi()
        return 1
    else:
        return 0

def func_1(n_ants, n_cities):
    global num_of_ants_x_colony, num_of_cities, positionsA, positionsB
    num_of_ants_x_colony = n_ants
    num_of_cities = n_cities
    positionsA = np.random.randint(num_of_cities+1, size=num_of_ants_x_colony)
    positionsB = np.random.randint(num_of_cities+1, size=num_of_ants_x_colony)

def func_2(l):
    global graph, num_of_cities
    graph.city_creator_random(num_of_cities,l)

def func_3():
    global map, q_0, alpha, beta, gamma, epsilon, rho, chi, t_o, phi
    map = Map(graph.mat_adj, q_0, alpha, beta, gamma, epsilon, rho, chi, t_o, phi)


def clean():
    global ants_totA,ants_totB,best_ant_distA, best_ant_distB, best_ant_dist_tourA, best_ant_dist_tourB
    ants_totA = []
    ants_totB = []
    best_ant_distA = []
    best_ant_distB = []
    best_ant_dist_tourA = []
    best_ant_dist_tourB = []

def iterator(n, update=False):
    global positionsA,positionsB, map, num_of_ants_x_colony, ants_totA, ants_totB, best_ant_distA, best_ant_distB, best_ant_dist_tourA, best_ant_dist_tourB
    for i in range(n):
        if update:
            positionsA = np.array([objectiveA[i] for i in np.random.randint(0, len(objectiveA), size=num_of_ants_x_colony)])
            positionsB = np.array([objectiveB[i] for i in np.random.randint(0, len(objectiveB), size=num_of_ants_x_colony)])
        antsA = [AntA(i,map,positionsA[i],1) for i in range(num_of_ants_x_colony)]
        antsB = [AntB(i,map,positionsB[i],2) for i in range(num_of_ants_x_colony)]
        map.set_ants(antsA, antsB)
        map.iterator()
        ants_totA.append(map.best_antA)
        ants_totB.append(map.best_antB)
        best_ant_distA.append(map.best_antA.current_distance)
        best_ant_distB.append(map.best_antB.current_distance)
        best_ant_dist_tourA.append(map.best_antA.tour)
        best_ant_dist_tourB.append(map.best_antB.tour)
        #pheromones_tot.append(map.get_max_pheromone_tour())

def find_best_path():
    global ants_totA,ants_totB,best_ant_distA,best_ant_distB
    best = 10000000000
    index = 0
    distanceInter = graph.find_intersect_distance(best_ant_dist_tourA, best_ant_dist_tourB)
    for i in range(len(ants_totA)):
        if best_ant_distA[i]+best_ant_distB[i]-distanceInter[i] < best:
            best = best_ant_distA[i]+best_ant_distB[i]
            index = i

    return best,index

def plot():
    global graph, map, ants_totA, ants_totB, best_ant_dist_tourA, best_ant_dist_tourB, best_ant_distA, best_ant_distB
    print("Puntos")
    graph.plot_points()
    print("Tour stocastic")
    val, index = find_best_path()
    graph.plot_final_tours(ants_totA[index],ants_totB[index] ,edges=False)
    distanceInter = graph.find_intersect_distance(best_ant_dist_tourA, best_ant_dist_tourB)
    graph.plot_distances_N(best_ant_distA,best_ant_distB,distanceInter)
    print("Pheromones in rounds")
    graph.plot_final_pheromones(map.pheromonesA.history, edges=False, title="A ")
    graph.plot_final_pheromones(map.pheromonesB.history, edges=False, title="B ")
    graph.plot_final_all_torus(ants_totA, ants_totB, edges=False)
    #graph.plot_max_pheromone_tours(pheromones_tot,edges=False)

def demo():
    global objectiveA, objectiveB
    rt = input("En el caso de usar de usar el demo, desea usar uno de los predeterminados? ")
    if rt == "si":
        eleccion = input("cual? (1,2,3) ")
        if eleccion == "1":
            map.change_objectiveA(objectiveA1)
            map.change_objectiveB(objectiveB1)
            objectiveA = objectiveA1
            objectiveB = objectiveB1

        if eleccion == "2":
            map.change_objectiveA(objectiveA2)
            map.change_objectiveB(objectiveB2)
            objectiveA = objectiveA2
            objectiveB = objectiveB2

        if eleccion == "3":
            map.change_objectiveA(objectiveA3)
            map.change_objectiveB(objectiveB3)
            objectiveA = objectiveA3
            objectiveB = objectiveB3

        if eleccion == "4":
            map.change_objectiveA(objectiveA4)
            map.change_objectiveB(objectiveB4)
            objectiveA = objectiveA4
            objectiveB = objectiveB4
    else:
        print("jajaja casi")

def setup():
    r1 = input("¿Desea trabajar con el grafo por defecto? (si/no) \n Respuesta: ")
    if r1 == "no":
        n_cities= int(input("Ingrese el número de ciudades.\n Respuesta: "))
        n_ants= int(input("Ingrese el número de hormigas por colonia.\n Respuesta: "))
        func_1(n_ants, n_cities)
        l = 1
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
    rk = input("¿Desea cambiar los objetivos? ")
    if rk == "si":
        demo()
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


