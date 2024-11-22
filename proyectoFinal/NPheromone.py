import numpy as np
from Map import *
from Ant import *

class NPheromone:

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
        pass
        
    # Second update
    def update_local(self):
        pass

class NPheromonesA(NPheromone):
    # First update
    def update_gb(self):
        L_gb, ant_gb = self.map.global_best_tourA()
        i, j = ant_gb.tour[-2], ant_gb.tour[-1]
        self.pheromones_matrix[i, j] = (1 - self.rho)*self.pheromones_matrix[i, j] + self.rho * (1/L_gb)
        self.pheromones_matrix[j, i] = (1 - self.rho)*self.pheromones_matrix[j, i] + self.rho * (1/L_gb)

        # Second update
    def update_local(self):
        for ant in self.map.antsA:
            i, j = ant.tour[-2], ant.tour[-1]
            self.pheromones_matrix[i, j] = (1 - self.chi)*self.pheromones_matrix[i, j] + self.chi * self.t0
            self.pheromones_matrix[j, i] = (1 - self.chi)*self.pheromones_matrix[j, i] + self.chi * self.t0

class NPheromonesB(NPheromone):
    # First update
    def update_gb(self):
        L_gb, ant_gb = self.map.global_best_tourB()
        i, j = ant_gb.tour[-2], ant_gb.tour[-1]
        self.pheromones_matrix[i, j] = (1 - self.rho)*self.pheromones_matrix[i, j] + self.rho * (1/L_gb)
        self.pheromones_matrix[j, i] = (1 - self.rho)*self.pheromones_matrix[j, i] + self.rho * (1/L_gb)

        # Second update
    def update_local(self):
        for ant in self.map.antsB:
            i, j = ant.tour[-2], ant.tour[-1]
            self.pheromones_matrix[i, j] = (1 - self.chi)*self.pheromones_matrix[i, j] + self.chi * self.t0
            self.pheromones_matrix[j, i] = (1 - self.chi)*self.pheromones_matrix[j, i] + self.chi * self.t0

        