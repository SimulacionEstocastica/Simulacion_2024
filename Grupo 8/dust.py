import numpy as np
import ust


def rescalate(vertexs):
    re_vers = []
    for i in vertexs:
        v1 = 4*i+2
        re_vers.append(v1)
    return re_vers


class Dualgraph:
    """
    Clase que a partir de un grafo de paralelepipedo regular,
    genera una base para hacer el dual.

    Attributes:
        shape (tuple[int]): Dimensionalidad del grafo base a generar.
        grid (np.array(boolean)): Malla usada como la base
        graph (Grafo): Grafo base.

    """
    def __init__(self, shape, start=None):
        """
        Attributes:
            shape (tuple(int)): Dimensionalidad del grafo dual.
            start (int, optional): Nodo razo para hacer el arbol.
        """
        # Se genera el grafo original
        g = ust.Grafo(shape, start)
        fil, col = shape
        # Se genera el grafo que permita recorrer los elementos
        self.shape = (4*fil+1, 4*col+1)
        self.grid = np.zeros(((4*fil+1, 4*col+1)))
        self.graph = g
        self.actives = []
        self.path = []

    def wilson(self):
        s = self.graph.wilson()
        for i in range(len(s)):
            if i%2 == 1:
                self.path.append(s[i])

    def append(self, vertex):
        """
        Método que agrega nodo al conjunto de los conectados a la raíz

        Params:
            vertex (tuple[int]): Tupla de dos elementos

        Return:
            None
        """
        # Verifica que sea un vertice perteneciente al grafo
        
        assert self.isVertex(vertex)
        # Le cambia el valor a 1 para que sea reconocido como 1 vertice parte
        # del arbol de raíz
        if len(vertex.shape) > 1:
            self.grid[vertex[:, 0], vertex[:, 1]] = 1
        else:
            self.grid[vertex[0], vertex[1]] = 1

    def isVertex(self, vertex):
        """
        Método que comprueba que la tupla elegida pertenezca al grafo

        Params:
            vertex (tuple[int]): Tupla de dos elementos.

        Return:
            Boolean
        """
        if isinstance(vertex, np.ndarray):
            if len(vertex.shape) > 1:
                a = (0 <= vertex[:, 0]).all()
                b = (vertex[:, 0] < self.shape[0]).all()
                c = (0 <= vertex[:, 1]).all()
                d = (vertex[:, 1] < self.shape[1]).all()
                return a and b and c and d
        a = (0 <= vertex[0])
        b = vertex[0] < self.shape[0]
        c = (0 <= vertex[1])
        d = (vertex[1] < self.shape[1])
        return a and b and c and d

    def scalate(self):
        paths = self.path
        for i in paths:
            # ahora comienza con el reescalamiento
            a = len(i)
            b = rescalate(i)
            for j in range(0, a-1):
                first = b[j]
                fifth = b[j+1]
                movs = np.array((fifth - first)/4, dtype=int)
                second = np.array(first + movs, dtype=int)
                third = np.array(first + 2*movs, dtype=int)
                fourth = np.array(first + 3*movs, dtype=int)
                self.actives.append(first)
                self.actives.append(second)
                self.actives.append(third)
                self.actives.append(fourth)
                if j == a-2:
                    self.actives.append(fifth)
                
    def gridact(self):
        for i in self.actives:
            self.append(i)

    def adyacent(self, path):
        directions = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ])
        if isinstance(path, np.ndarray) and len(path.shape) > 1:
            neighbors = path[:, np.newaxis, :] + directions
            neighbors = neighbors.reshape(-1, 2)
        else:
            neighbors = path + directions

        valid_neighbors = neighbors[np.apply_along_axis(self.isVertex, 1, neighbors)]
        return np.unique(valid_neighbors, axis=0)

    def dual(self):
        adj = self.adyacent(np.array(self.actives))
        M = np.zeros(self.shape)
        M[adj[:, 0], adj[:, 1]] = 1
        M = 1 - M
        return M
