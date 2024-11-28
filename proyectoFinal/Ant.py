import Pheromone as ph
from Map import *
import numpy as np

class Ant:
    def __init__(self, id, map, current_city):
        # ID
        self.id = id
        # Map
        self.map = map
        # Length tour
        self.current_distance = 0
        # Current city
        self.current_city = current_city
        # Current tour
        self.tour = [self.current_city]

    
    def update_tour_dist(self, j):
        self.current_distance += self.map.distance_ij(self.current_city, j)
        self.tour.append(j)
        self.current_city = j

    def choice_city(self):
        # coin==1 maximo de ...
        # coin==0 distribucion dada
        coin = np.random.binomial(1, self.map.q_0)
        info_neighbor = self.map.info_neighbor(self.current_city)
        pher_dist_index = [(pheromones,distances,i) for (pheromones,distances,i) in info_neighbor if i not in self.tour]
        if len(pher_dist_index) == 0:
            return self.tour[0]
        if coin == 0:
            b = [(t_il**(self.map.alpha) * (1/d_il)**(self.map.beta), l) for t_il, d_il, l in pher_dist_index]
            c = sum(x[0] for x in b)
            unif = np.random.uniform(0, c)
            acc = b[0][0]
            j = 0
            while unif > acc:
                j += 1
                acc += b[j][0]
            return b[j][1]
        else:
            a = [(t_il * (1/d_il)**(self.map.beta), l) for t_il, d_il, l in pher_dist_index]
            return max(a, key=lambda x: x[0])[1]


    
    



