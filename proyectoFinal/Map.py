import Pheromone
import Ant
import logging as lg

lg.basicConfig(level=lg.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Map:
    def __init__(self, matrix, q_0, alpha, beta,rho,chi,t_o):
        # Adjacency matrix
        self.adjacency_matrix = matrix
        # Ants
        self.ants = []
        # Best ant
        self.best_ant = 0
        # Alpha
        self.alpha = alpha
        # Beta
        self.beta = beta
        # Number of cities
        self.N = self.adjacency_matrix.shape[0]
        # probability of chosing max pheromone-distance path
        self.q_0 = q_0
        # Pheromones
        self.pheromones = Pheromone.Pheromone(matrix.shape[0], self, rho, chi, t_o)
    
    def set_ants(self, ants):
        self.ants = ants
        
    def distance_ij(self, i, j):
        return self.adjacency_matrix[i,j]
    
    def distance_path(self, l):
        n = len(l)
        total_distance = 0
        for i in range(n-1):
            total_distance += self.adjacency_matrix[l[i], l[i + 1]]
        return total_distance

    def global_best_tour(self):
        distances = [(ant.current_distance, ant) for ant in self.ants]
        minimum = min(distances, key=lambda x: x[0])
        self.best_ant = minimum[1]
        return minimum
    
    def info_neighbor(self, i):
        distances_neighbor = self.adjacency_matrix[i,:]
        pheromones_neighbor = self.pheromones.pheromones_matrix[i,:]
        return list(zip(pheromones_neighbor, distances_neighbor, range(self.N)))

    #La siguiente función tiene como fin hacer una iteracion
    #parámetros: ninguno
    #return: ninguno
    #Esta función le preguntará a cada una de sus hormigas hacer un paso, luego actualizará las feromonas

    def iteration(self):
        for ant in self.ants:
            lg.debug("actualizando la hormiga: "+str(ant.id))
            choice = ant.choice_city()
            lg.debug("siguiente ciudad de la hormiga: "+str(choice))
            ant.update_tour_dist(choice)
            lg.debug("Su tour actual: "+str(ant.tour))
            lg.debug("Su distancia recorrida: "+str(ant.current_distance))

        lg.debug("actualizando feromonas globalmente")
        self.pheromones.update_gb()
        lg.debug("actualizando feromonas localmente")
        self.pheromones.update_local()
        #self.pheromones.history.append(self.pheromones.pheromones_matrix.copy())
        lg.debug("Las feromonas son : \n"+str(self.pheromones.pheromones_matrix))
    

    # La siguiente función tiene como fin correr la simulación n veces
    # parámetros: ninguno
    # return: ninguno
    # Esta función le preguntará a cada una de sus hormigas hacer un paso, luego actualizará las feromonas

    def iterator(self):
        for _ in range(self.N):
            self.iteration()
        self.pheromones.history.append(self.pheromones.pheromones_matrix.copy())
        lg.debug("La mejor ruta es la de la hormiga "+str(self.best_ant.id))
        lg.debug("Su ruta es "+str(self.best_ant.tour))
        lg.debug("Su recirrudo tiene distancia "+str(self.best_ant.current_distance))
        lg.debug("las feromonas son: \n"+str(self.pheromones.pheromones_matrix))

    def get_max_pheromone_tour(self):
        return self.pheromones.max_pheromone_tour()
    

    

    
    



        
