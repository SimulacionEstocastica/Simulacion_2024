import numpy as np
import sys
import os
# Add the correct path to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ust import erase_loops, Grafo
import unittest


class TestGrafo(unittest.TestCase):
    def setUp(self):
        self.grafo_1 = Grafo(start=np.array([5, 5]))
        self.grafo_2 = Grafo((10, 3), start=np.array([9, 2]))
        return super().setUp()

    def testDefaultShape(self):
        '''
        Prueba que las dimensiones por defecto de la grilla
        son (10, 10)
        '''
        self.assertEqual(self.grafo_1.shape, (10, 10))
        self.assertEqual(self.grafo_2.shape, self.grafo_2.grid.shape)

    def testShape(self):
        '''
        Prueba que la forma del grafo se puede fijar correctamente usando
        el constructor
        '''
        self.assertEqual(self.grafo_2.shape,  (10, 3))
        self.assertEqual(self.grafo_2.shape, self.grafo_2.grid.shape)

    def testStart(self):
        '''
        Prueba que el grafo puede inicializarse correctamente
        con el vértice start
        '''
        test_grid = np.zeros((10, 10))
        test_grid[5, 5] = 1
        self.assertTrue((test_grid == self.grafo_1.grid).all())
        test_grid2 = np.zeros((10, 3))
        test_grid2[9, 2] = 1
        self.assertTrue((test_grid2 == self.grafo_2.grid).all())

    def testAppend(self):
        '''
        Prueba que se puedan añadir vertices correctamente al grafo
        '''
        test_grid = np.zeros((10, 10))
        test_grid[5, 5] = 1
        verts = np.array([[n, 0] for n in range(10)])
        test_grid[:, 0] = np.ones(10)
        self.grafo_1.append(verts)
        self.assertTrue((test_grid == self.grafo_1.grid).all())

    def testRandomSucc(self):
        '''
        Prueba que RandomSuccesor retorna un vértice válido
        '''
        s, _ = self.grafo_1.random_succesor(np.array([3, 3]))
        self.assertEqual(np.linalg.norm(s-np.array([3, 3])), 1)
        self.assertTrue(self.grafo_1.isVertex(s))

        t, _ = self.grafo_2.random_succesor(np.array([0, 0]))
        self.assertEqual(np.linalg.norm(t-np.array([0, 0])), 1)
        self.assertTrue(self.grafo_2.isVertex(t))

    def testRandomWalk(self):
        '''
        Prueba que RandomWalk retorna un arreglo de vertices válidos con distancia 1 entre sí
        y que el camino termina en un vertice del arbol
        '''
        for _ in range(50):
            rw, _ = self.grafo_1.random_walk(np.array([0, 0]))
            self.assertTrue(self.grafo_1.isVertex(rw))
            self.assertTrue((np.linalg.norm(np.diff(rw, axis=0), axis=1) == 1).all())
            self.assertTrue((rw[-1] == np.array([5, 5])).all())

    def testEraseLoops(self):
        '''
        Prueba que EraseLoops retorna un camino sin loops
        '''
        for _ in range(100):
            visited, dirs = self.grafo_1.random_walk(np.random.randint(0, 10, 2))
            lerw, dir = erase_loops(visited, dirs)
            nodos, count = np.unique(lerw, return_counts=True, axis=0)
            self.assertTrue(not (count > 1).any())
            self.assertTrue(self.grafo_1.isVertex(lerw))
            self.assertTrue((np.linalg.norm(np.diff(lerw, axis=0), axis=1) == 1).all())
            self.assertTrue((lerw[-1] == np.array([5, 5])).all())

    def testWilson(self):
        paths = self.grafo_1.wilson()
        self.assertTrue((self.grafo_1.grid == 1).all())
        for i, path in enumerate(paths):
            if i % 2 == 1:
                nodos, count = np.unique(path, return_counts=True, axis=0)
                self.assertTrue((count == 1).all())


unittest.main()
