import unittest
from Map import *
from Ant import *
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_constructor(self):
        #self.assertEqual(True, False)  # add assertion here
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        self.assertTrue(np.array_equal(a.adjacency_matrix,np.array([[0,1],[2,0]])))
        self.assertEqual(a.ants , [])
        self.assertEqual(a.best_ant, 0)
        self.assertEqual(a.alpha, 1)
        self.assertEqual(a.beta, 1)
        self.assertEqual(a.N , 2)
        self.assertEqual(a.q_0 , 0.5)

    def test_set_ants(self):
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        ant = Ant(1, a, 1)
        a.set_ants([ant])
        self.assertEqual(a.ants, [ant])
        ant2 = Ant(2, a, 0)
        a.set_ants([ant,ant2])
        self.assertEqual(a.ants, [ant,ant2])

    def test_distance_ij(self):
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        self.assertEqual(a.distance_ij(0,0), 0)
        self.assertEqual(a.distance_ij(0,1), 1)
        self.assertEqual(a.distance_ij(1,0), 2)
        self.assertEqual(a.distance_ij(1,1), 0)

    def test_distance_path(self):
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        res = a.distance_path([0,1])
        self.assertEqual(1,res)
        self.assertEqual(a.distance_path([0,1,0]), 3)

    def test_global_best_tour(self):
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        ant = Ant(1, a, 1)
        ant2 = Ant(2, a, 0)
        a.set_ants([ant, ant2])
        ant.update_tour_dist(0)
        ant2.update_tour_dist(1)
        res = a.global_best_tour()
        self.assertEqual(res, (1,ant2))
        self.assertEqual(a.best_ant, ant2)

    def test_info_neighbor(self):
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        res0 = [(0, 0, 0), (0, 1, 1)]
        self.assertEqual(res0,a.info_neighbor(0))
        res1 = [(0, 2, 0), (0, 0, 1)]
        self.assertEqual(res1,a.info_neighbor(1))




if __name__ == '__main__':
    unittest.main()
