import NPheromone
import NAnt
import logging as lg

lg.basicConfig(level=lg.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Map:
    def __init__(self, matrix, q_0, alpha, beta, gamma, epsilon, rho, chi, t_o, phi):
        # Adjacency matrix
        self.adjacency_matrix = matrix
        # Ants A
        self.antsA = []
        # Ants B
        self.antsB = []
        # Best ant A
        self.best_antA = []
        # Best ant B
        self.best_antB = []
        # Alpha
        self.alpha = alpha
        # Beta
        self.beta = beta
        # Gamma
        self.gamma = gamma
        # Epsilon
        self.epsilon = epsilon
        # Penalty
        self.phi = phi
        # Number of cities
        self.N = self.adjacency_matrix.shape[0]
        # probability of chosing max pheromone-distance path
        self.q_0 = q_0
        # Pheromones A
        self.pheromonesA = NPheromone.NPheromonesA(matrix.shape[0], self, rho, chi, t_o)
        # Pheromones B
        self.pheromonesB = NPheromone.NPheromonesB(matrix.shape[0], self, rho, chi, t_o)
        # Objective A
        self.objectiveA = range(self.N)
        # Objective B
        self.objectiveB = range(self.N)
    
    def set_ants(self, antsA, antsB):
        self.antsA = antsA
        self.antsB = antsB
        
    def distance_ij(self, i, j):
        return self.adjacency_matrix[i,j]
    
    def distance_path(self, l):
        n = len(l)
        total_distance = 0
        if n!=0:
            for i in range(n-1):
                total_distance += self.adjacency_matrix[l[i], l[i + 1]]
        return total_distance

    def global_best_tourA(self):
        distances = [(ant.current_distance, ant) for ant in self.antsA]
        minimum = min(distances, key=lambda x: x[0])
        self.best_antA = minimum[1]
        return minimum
    
    def global_best_tourB(self):
        distances = [(ant.current_distance, ant) for ant in self.antsB]
        minimum = min(distances, key=lambda x: x[0])
        self.best_antB = minimum[1]
        return minimum
    
    def info_neighbor(self, i):
        distances_neighbor = self.adjacency_matrix[i,:]
        pheromones_neighborA = self.pheromonesA.pheromones_matrix[i,:]
        pheromones_neighborB = self.pheromonesB.pheromones_matrix[i,:]

        return list(zip(pheromones_neighborA, pheromones_neighborB, distances_neighbor, range(self.N)))
    
    def change_objectiveA(self, cities):
        self.objectiveA = cities

    def change_objectiveB(self, cities):
        self.objectiveB = cities

    #La siguiente función tiene como fin hacer una iteracion
    #parámetros: ninguno
    #return: ninguno
    #Esta función le preguntará a cada una de sus hormigas hacer un paso, luego actualizará las feromonas

    def iteration(self):
        for ant in self.antsA:
            if ant.tour[-1] not in ant.tour[:-1]:
                lg.debug("actualizando la hormiga: "+str(ant.id))
                choice = ant.choice_city()
                lg.debug("siguiente ciudad de la hormiga: "+str(choice))
                ant.update_tour_dist(choice)
            else:
                ant.clean()
            lg.debug("Su tour actual: "+str(ant.tour))
            lg.debug("Su distancia recorrida: "+str(ant.current_distance))
        for ant in self.antsB:
            if ant.tour[-1] not in ant.tour[:-1]:
                lg.debug("actualizando la hormiga: "+str(ant.id))
                choice = ant.choice_city()
                lg.debug("siguiente ciudad de la hormiga: "+str(choice))
                ant.update_tour_dist(choice)
            else:
                ant.clean()
            lg.debug("Su tour actual: "+str(ant.tour))
            lg.debug("Su distancia recorrida: "+str(ant.current_distance))
        lg.debug("actualizando feromonas A globalmente")
        self.pheromonesA.update_gb()
        lg.debug("actualizando feromonas A localmente")
        self.pheromonesA.update_local()
        lg.debug("Las feromonas son : \n"+str(self.pheromonesA.pheromones_matrix))

        lg.debug("actualizando feromonas B globalmente")
        self.pheromonesB.update_gb()
        lg.debug("actualizando feromonas B localmente")
        self.pheromonesB.update_local()
        lg.debug("Las feromonas son : \n"+str(self.pheromonesB.pheromones_matrix))
    

    # La siguiente función tiene como fin correr la simulación n veces
    # parámetros: ninguno
    # return: ninguno
    # Esta función le preguntará a cada una de sus hormigas hacer un paso, luego actualizará las feromonas

    def iterator(self):
        for _ in range(self.N):
            self.iteration()

        self.pheromonesA.history.append(self.pheromonesA.pheromones_matrix.copy())
        self.pheromonesB.history.append(self.pheromonesB.pheromones_matrix.copy())
        lg.debug("La mejor ruta del bando A es la de la hormiga "+str(self.best_antA.id))
        lg.debug("Su ruta es "+str(self.best_antA.tour))
        lg.debug("Su recirrudo tiene distancia "+str(self.best_antA.current_distance))
        lg.debug("las feromonas son: \n"+str(self.pheromonesA.pheromones_matrix))

        lg.debug("La mejor ruta del bando B es la de la hormiga "+str(self.best_antB.id))
        lg.debug("Su ruta es "+str(self.best_antB.tour))
        lg.debug("Su recirrudo tiene distancia "+str(self.best_antB.current_distance))
        lg.debug("las feromonas son: \n"+str(self.pheromonesB.pheromones_matrix))
   