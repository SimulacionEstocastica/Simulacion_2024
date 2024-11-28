import unittest
import numpy as np
from Graphicator import Graphicator


class MyTestCase(unittest.TestCase):
    def test_city_creator_random(self):
        a = Graphicator()
        a.city_creator_random(2,1)
        points = a.points
        mat_adj = a.mat_adj
        for i in range(len(points)):
            for j in range(len(points)):
                self.assertEqual(np.linalg.norm(points[i] - points[j]), mat_adj[i][j])

    def test_city_creator_random(self):
        a = Graphicator()
        a.city_creator(np.array([(0,1),(1,1)]))
        self.assertTrue(np.array_equal(np.array([[0,1],[1,0]],a.mat_adj)))


if __name__ == '__main__':
    unittest.main()
