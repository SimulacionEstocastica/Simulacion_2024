import unittest

import numpy as np

from Ant import Ant
from Map import Map
from Pheromone import Pheromone


class MyTestCase(unittest.TestCase):
    def test_constructor(self):
        a = Map(np.array([[0, 1], [2, 0]]), 0.5, 1, 1, 0.8, 0.7, 0.2)
        pheromones = a.pheromones
        self.assertTrue(np.array_equal(pheromones.pheromones_matrix , np.array([[0, 0], [0, 0]])))
        self.assertEqual(pheromones.rho , 0.8)
        self.assertEqual(pheromones.map , a)
        self.assertEqual(pheromones.chi , 0.7)
        self.assertEqual(pheromones.t0 , 0.2)
    def test_pheromone_ij(self):
        a = Map(np.array([[0, 1], [2, 0]]), 0.5, 1, 1, 0.8, 0.7, 0.2)
        pheromones = a.pheromones
        self.assertEqual(pheromones.pheromone_ij(0,1) , 0)
        self.assertEqual(pheromones.pheromone_ij(1,0) , 0)
        #I guess we are doing a bit of cheating (otherwise testing won't work), so suppose the pheromone matrix is as follows
        pheromones.pheromones_matrix = np.array([[0,5],[10,0]])
        self.assertEqual(pheromones.pheromone_ij(0,1) , 5)
        self.assertEqual(pheromones.pheromone_ij(1,0) , 10)

    def test_update_gb(self):
        a = Map( np.array([[0,1,2],[1,0,3],[2,3,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        pheromones = a.pheromones
        ant = Ant(1, a, 1)
        ant2 = Ant(2, a, 0)
        a.set_ants([ant, ant2])
        ant.update_tour_dist(0)
        ant2.update_tour_dist(1)
        tour = [1, 0]
        pheromones.update_gb()
        self.assertEqual(pheromones.pheromones_matrix[0, 1],  0.8)
        self.assertEqual(pheromones.pheromones_matrix[1, 0],  0.8)

    def test_update_local(self):
        a = Map(np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), 0.5, 1, 1, 0.8, 0.7, 0.2)
        pheromones = a.pheromones
        ant = Ant(1, a, 1)
        ant2 = Ant(2, a, 2)
        a.set_ants([ant, ant2])
        ant.update_tour_dist(0)
        curr_city_1 =  ant.current_city
        ant2.update_tour_dist(1)
        curr_city_2 =  ant2.current_city
        tour1 = [1, 0]
        tour2 = [2, 1]
        pheromones.update_local()
        self.assertEqual(pheromones.pheromones_matrix[1, 0], 0.7*0.2)
        self.assertEqual(pheromones.pheromones_matrix[0, 1], 0.7*0.2)
        self.assertEqual(pheromones.pheromones_matrix[2, 1], 0.7*0.2)
        self.assertEqual(pheromones.pheromones_matrix[1, 2], 0.7*0.2)

    def max_pheromone_tour(self):
        a = Map(np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), 0.5, 1, 1, 0.8, 0.7, 0.2)
        pheromones = a.pheromones
        ant = Ant(1, a, 1)
        ant2 = Ant(2, a, 2)
        a.set_ants([ant, ant2])
        ant.update_tour_dist(0)
        curr_city_1 =  ant.current_city
        ant2.update_tour_dist(1)
        curr_city_2 =  ant2.current_city
        tour1 = [1, 0]
        tour2 = [2, 1]
        pheromones.update_local()
        print(pheromones.max_pheromone_tour())


if __name__ == '__main__':
    unittest.main()
