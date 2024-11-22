import numpy as np


direcciones = {
    "u": np.array([0, 1]),
    "d": np.array([0, -1]),
    "l": np.array([-1, 0]),
    "r": np.array([1, 0])
}

str_dir = np.array(["u", "d", "l", "r"])


def ordered_scan(arr, set):
    """
    Revisa el arreglo arr ordenadamente hasta que encuentre un
    elemento contenido en set

    Parámetros
    -----------
    arr : numpy.array
        Arreglo a escanear

    set: iterable
        Conjunto de elementos a buscar

    Retorna
    -----------
    output :
        Primer elemento encontrado
    """
    for i in range(len(arr)):
        if np.equal(set, arr[i]).all(axis=1).any():
            return arr[i]


# la siguiente función fue optimizada utilizando Chatgpt
def erase_loops(visited, dirs):
    """
    Borra los loops de una caminata aleatoria dada.

    Parámetros
    -----------
    visited: numpy.ndarray
        Caminata como lista de vértigices vecinos en el grafo.

    dirs: numpy.ndarray
        Lista de direcciones tomadas en la realización del camino.

    Retorna
    ----------
    loop_erased: numpy.array[int]
        Caminata sin loops.

    new_dirs: numpy.array[str]
        Direcciones del camino sin loops.
    """
    # Diccionario para almacenar la última posición de cada vértice
    seen = {}
    loop_erased = []
    new_dirs = []

    for i, v in enumerate(visited):
        vt = tuple(v)  # Convertir el vértice a una tupla (hashable)
        if vt in seen:
            # Si el vértice ya fue visitado, borrar el loop
            loop_start = seen[vt]
            loop_erased = loop_erased[:loop_start]
            new_dirs = new_dirs[:loop_start]
            seen = {tuple(v): idx for idx, v in enumerate(loop_erased)}
        loop_erased.append(v)
        if i < len(dirs):  # Asegurarse de no salir del rango
            new_dirs.append(dirs[i])
        seen[vt] = len(loop_erased) - 1  # Actualizar índice más reciente
    return np.array(loop_erased), np.array(new_dirs)


class Grafo:
    """
    Clase que representa un grafo, usada como base para realizar el algoritmo
    de Wilson y una implementación particular.

    Params:
        shape (tuple[int]): Dimensionalidad del grafo a generar.
        start (int, optional): Nodo raíz inicial del grafo.
    """
    def __init__(self, shape=(10, 10), start=None):
        self.shape = shape
        self.grid = np.zeros(shape)
        # En caso de que el nodo inicial haya sido elegido
        # se le aigna valor 1 para decir que ya fue visitado
        if start is not None:
            self.st = start
            self.grid[tuple(start)] = 1
        # si no, se asigna uno de forma aleatoria entre todos los disponibles
        else:
            i = np.random.randint(0, shape[0])
            j = np.random.randint(0, shape[1])
            self.st = [i, j]
            self.grid[i, j] = 1

    def get_start(self):
        return self.st

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
        # Le cambia el valor a 1 para que sea reconocido como
        # un vertice parte del arbol de raíz
        if len(vertex.shape) > 1:
            # for vertice in vertex:
            #     self.grid[tuple(vertice)] = 1
            self.grid[vertex[:, 0], vertex[:, 1]] = 1
        else:
            self.grid[tuple(vertex)] = 1

    def random_succesor(self, vertex):
        """
        Método que genera el nodo siguiente del actual

        Params:
            vertex (tuple[int]): Tupla de dos elementos

        Return:
            vertex: el nodo que sigue del inicial
            dir: la dirección en la cual se movió el inicial para
            llegar al actual
        """
        # se elige la dirección de forma aleatoria entre las 4 posibles
        dir = str_dir[np.random.randint(0, 4)]
        # se generan posibles sucesores hasta tener un nodo válido
        while not self.isVertex(vertex+direcciones[dir]):
            dir = str_dir[np.random.randint(0, 4)]
        return vertex+direcciones[dir], dir

    def random_walk(self, start):
        """
        Metodo que genera un paseo aleatorio desde un nodo que no pertenece al
        árbol actual

        Params:
            start (tuple[int]): Tupla de dos elementos

        Return:
            camino(array[int]): los nodos del camino.
            direcciones(array(int)): las direcciones en las que se
            mueve para recorrer el camino
        """
        # se genera una lista para almacenar los elementos visitados
        # iniciando con el nodo entregado
        visited = [start]
        # se genera la lista para guardar las direcciones
        dirs = []
        # mientras no se llegue a un elemento que ya pertenece al árbol
        # se sigue ejecutando
        while True:
            # nodo actual es el último de la lista
            current = visited[len(visited)-1]
            # se utiliza random_succesor para obtener el siguiente
            next, dir = self.random_succesor(current)
            # se añade el nodo siguiente y
            # la dirección a la lista correspondiente
            visited.append(next)
            dirs.append(dir)
            # si se llega a uno que pertenece al arbol se rompe el ciclo
            if self.grid[tuple(visited[-1])]:
                break
        # se le añaden nombres a las listas
        camino = np.array(visited, dtype=int)
        direcciones = np.array(dirs, dtype=str)
        return camino, direcciones

    def wilson(self):
        """
            Genera un árbol a partir de la raíz

            Retorna
            ----------
            paseos: numpy.array[np.array[int]]
                Lista de paseos y caminos realizados para generar el árbol
        """
        # Se inicializa la lista
        paseos = []
        # Mientras existan puntos los cuales no pertenezcan, se sigue iterando
        while (self.grid != np.ones(self.shape)).any():
            # Se obtienen los vértices que aún no pertenecen al árbol
            out = np.array(np.nonzero(1 - self.grid))
            # Se genera de forma uniforme un nuevo punto de partida
            length = out.shape[1]
            nstart = out[:, np.random.randint(0, length)]
            # Se realiza la primera iteración
            cicled, dired = self.random_walk(nstart)
            # Se agrega el paseo con loops
            paseos.append(cicled)
            # Se le quitan los loops
            nocicled = erase_loops(cicled, dired)[0]
            # Se añade sin ciclo al grid
            self.append(nocicled)
            # Se agrega el camino
            paseos.append(nocicled)
        # Se retorna
        return paseos
