import sys, os
import numpy as np
import unittest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dust import Dualgraph


class TestGrafoDual(unittest.TestCase):
    def setUp(self):
        self.d_1 = Dualgraph((10, 10))
        self.d_2 = Dualgraph((7, 19))

    def testShape(self):
        self.assertEqual(self.d_1.graph.shape, (10, 10))
        self.assertEqual(self.d_2.graph.shape, (7, 19))
        self.assertEqual(self.d_1.shape, (41, 41))
        self.assertEqual(self.d_2.shape, (29, 77))

    def testAppend(self):
        verts = np.array([[0, n] for n in range(41)])
        self.assertTrue((self.d_1.grid == 0).all())
        self.d_1.append(verts)
        self.assertTrue((self.d_1.grid[0, :] == 1).all())
        self.assertTrue((self.d_1.grid != 1).any())

    # def testReescalate(self):
    #     self.d_1.reescalategraph()
    #     self.assertTrue((self.d_1.graph.grid == 1).all())
    #     self.d_2.reescalategraph()
    #     self.assertTrue((self.d_2.graph.grid == 1).all())

    def testAdjacent(self):
        # Caso con un vértice
        path = np.array([1, 1])
        result = self.d_1.adyacent(path)
        expected = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ]) + 1
        np.testing.assert_array_equal(np.sort(result, axis=0),
                                      np.sort(expected, axis=0))

        # Se prueba que funcione para un camino de mas de un vértice
        path = np.array([[0, 0], [1, 1]])
        result = self.d_1.adyacent(path)
        expected = np.array([
            [1, 0], [0, 1],  # Neighbors of [0, 0]
            [2, 1], [0, 1], [1, 2], [1, 0],    # Neighbors of [1, 1]
            [1, 1],
            [2, 2], [2, 0], [0, 2], [0, 0],
        ])
        expected = np.unique(expected, axis=0)
        np.testing.assert_array_equal(np.sort(result, axis=0),
                                      np.sort(expected, axis=0))


if __name__ == "__main__":
    unittest.main()
