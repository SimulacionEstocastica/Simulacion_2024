import NPheromone as Nph
from NMap import *
import numpy as np

class Ant:
    def __init__(self, id, map, current_city, colony_id):
        # ID
        self.id = id
        # ID of colony
        self.id = colony_id
        # Map
        self.map = map
        # Length tour
        self.current_distance = 0
        # Epsilon
        self.epsilon = self.map.epsilon
        # Penalty
        self.phi = self.map.phi
        # Current city
        self.current_city = current_city
        # Current tour
        self.tour = [self.current_city]

    
    def update_tour_dist(self, j):
        self.current_distance += self.map.distance_ij(self.current_city, j)
        self.tour.append(j)
        self.current_city = j

    def choice_city(self):
        pass

    def clean(self):
        bucle = self.tour[-1]
        if bucle in self.tour[:-1]:
            index = 0
            for i in range(len(self.tour)):
                if self.tour[i] == bucle:
                    index = i
                    break
            self.distance = self.current_distance - self.map.distance_path(self.tour[0:index])
            self.tour = self.tour[index:]
        
class AntA(Ant):
    def choice_city(self):
        first_objective_city = -1
        index = 0
        pending_cities = [i for i in self.map.objectiveA if i not in self.tour]
        if len(pending_cities) == 0:
            for i, city in enumerate(self.tour):
                if city in self.map.objectiveA:
                    first_objective_city = city
                    index = i+1
                    break
            pending_cities = [first_objective_city]

        coin = np.random.binomial(1, self.map.q_0)
        info_neighbor = self.map.info_neighbor(self.current_city)
        pher_dist_index = [(pheromonesA, pheromonesB, distances,i) for (pheromonesA, pheromonesB, distances,i) in info_neighbor if i not in self.tour[index:]]
        def verf(i):
            if i in pending_cities:
                return 1
            else:
                return self.phi
        if len(pher_dist_index) == 0: 
            return self.tour[0]
        
        if coin == 0:
            b = [(verf(l)* (t_ilA+self.epsilon)**(self.map.alpha) * (1/(t_ilB+self.epsilon))**(self.map.gamma) * (1/d_il)**(self.map.beta), l) for t_ilA, t_ilB, d_il, l in pher_dist_index]


            c = sum(x[0] for x in b)
            unif = np.random.uniform(0, c)
            acc = b[0][0]
            j = 0

            while unif > acc:
                j += 1
                acc += b[j][0]

            return b[j][1]
        else:
            a = [(verf(l)* (t_ilA+self.epsilon) * (1/(t_ilB+self.epsilon)) * (1/d_il)**(self.map.beta), l) for t_ilA, t_ilB, d_il, l in pher_dist_index]

            return max(a, key=lambda x: x[0])[1]
        
class AntB(Ant):

    def choice_city(self):
        first_objective_city = -1
        index = 0
        pending_cities = [i for i in self.map.objectiveB if i not in self.tour]
        if len(pending_cities) == 0:
            for i, city in enumerate(self.tour):
                if city in self.map.objectiveB:
                    first_objective_city = city
                    index = i + 1
                    break
            pending_cities = [first_objective_city]

        coin = np.random.binomial(1, self.map.q_0)
        info_neighbor = self.map.info_neighbor(self.current_city)
        pher_dist_index = [(pheromonesA, pheromonesB, distances, i) for (pheromonesA, pheromonesB, distances, i) in
                           info_neighbor if i not in self.tour[index:]]

        def verf(i):
            if i in pending_cities:
                return 1
            else:
                return self.phi

        if len(pher_dist_index) == 0:
            return self.tour[0]

        if coin == 0:
            b = [(verf(l) * ((t_ilB+ self.epsilon)**(self.map.alpha)) * ((1 / (t_ilA + self.epsilon)) ** (self.map.gamma)) * ((
                        1 / d_il) ** (self.map.beta)), l) for t_ilA, t_ilB, d_il, l in pher_dist_index]

            c = sum(x[0] for x in b)
            unif = np.random.uniform(0, c)
            acc = b[0][0]
            j = 0

            while unif > acc:
                j += 1
                acc += b[j][0]

            return b[j][1]
        else:
            a = [(verf(l) * ((t_ilB+ self.epsilon)) * ((1 / (t_ilA + self.epsilon)) ) * ((
                        1 / d_il) ** (self.map.beta)), l) for
                 t_ilA, t_ilB, d_il, l in pher_dist_index]

            return max(a, key=lambda x: x[0])[1]