import numpy as np
import pulp

from Map import *
from Ant import *

class Pheromone:

    def __init__(self, n, map, rho, chi, t_o):
        # Pheromones adjacency matrix
        self.pheromones_matrix = np.zeros((n,n))
        # Parameter
        self.rho = rho
        # Map
        self.map = map
        # Chi
        self.chi = chi
        # t_0
        self.t0 = t_o
        # History
        self.history = [self.pheromones_matrix]
    
    def pheromone_ij(self, i, j):
        return self.pheromones_matrix[i,j]

    def pheromone_path(self, l):
        n = len(l)
        total_distance = 0
        for i in range(n-1):
            total_distance += self.pheromones_matrix[l[i], l[i+1]]

    # First update
    def update_gb(self):
        L_gb, ant_gb = self.map.global_best_tour()
        i, j = ant_gb.tour[-2], ant_gb.tour[-1]
        self.pheromones_matrix[i, j] = (1 - self.rho)*self.pheromones_matrix[i, j] + self.rho * (1/L_gb)
        self.pheromones_matrix[j, i] = (1 - self.rho)*self.pheromones_matrix[j, i] + self.rho * (1/L_gb)
        
    # Second update
    def update_local(self):
        for ant in self.map.ants:
            i, j = ant.tour[-2], ant.tour[-1]
            self.pheromones_matrix[i, j] = (1 - self.chi)*self.pheromones_matrix[i, j] + self.chi * self.t0
            self.pheromones_matrix[j, i] = (1 - self.chi)*self.pheromones_matrix[j, i] + self.chi * self.t0

    def max_pheromone_tour(self):
        n = len(self.pheromones_matrix[0])
        ph = -self.pheromones_matrix
        # Definir el modelo de optimización
        model = pulp.LpProblem("TSP", pulp.LpMinimize)

        # Variables de decisión: x[i, j] indica si se viaja de i a j
        x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(n) if i != j], cat="Binary")
        # Variables para la restricción de subtour
        u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n - 1, cat="Continuous")

        # Función objetivo: minimizar la distancia total
        model += pulp.lpSum(ph * x[i, j] for i in range(n) for j in range(n) if i != j)

        # Restricción: debe salir exactamente un arco de cada nodo
        for i in range(n):
            model += pulp.lpSum(x[i, j] for j in range(n) if i != j) == 1

        # Restricción: debe entrar exactamente un arco en cada nodo
        for j in range(n):
            model += pulp.lpSum(x[i, j] for i in range(n) if i != j) == 1

        # Restricciones de subtour para evitar ciclos
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model += u[i] - u[j] + n * x[i, j] <= n - 1

        # Resolver el modelo
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        # Obtener la solución óptima
        pre_tour = []
        if model.status == pulp.LpStatusOptimal:
            pre_tour = [(i, j) for i in range(n) for j in range(n) if i != j and pulp.value(x[i, j]) == 1]
        tour = [0]
        while len(tour) < n + 2:
            i = tour[-1]
            for edge in pre_tour:
                if edge[0] == i:
                    tour.append(edge[1])
        return tour
