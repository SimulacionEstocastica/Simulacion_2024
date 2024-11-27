import numpy as np


def dfsmatrix(M):
    # Encuentra índices donde M es igual a 0
    indices = np.transpose(np.nonzero(1 - M))
    # Selecciona un punto inicial aleatorio
    num = np.random.randint(0, len(indices))
    i, j = indices[num]
    # Direcciones de movimiento
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    # Inicializa la pila y estructuras de visitados y resultado
    pila = [(i, j)]
    visited = set()
    out = []

    # DFS iterativo
    while pila:
        curr = pila.pop()
        if curr not in visited:
            visited.add(curr)
            out.append(curr)

            for di, dj in directions:
                ni, nj = curr[0] + di, curr[1] + dj
                # Validar límites y evitar nodos ya visitados
                if (
                    0 <= ni < M.shape[0]
                    and 0 <= nj < M.shape[1]
                    and M[ni, nj] == 0
                    and (ni, nj) not in visited
                ):
                    pila.append((ni, nj))

    return np.array(out)

def erase_loops(visited, dirs):
    """
    Borra los loops de una caminata aleatoria dada.

    Parámetros
    -----------
    visited: numpy.array[int]
        Caminata como lista de vértices vecinos en el grafo.

    dirs: numpy.array[str]
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
        loop_erased.append(v)
        if i < len(dirs):  # Asegurarse de no salir del rango
            new_dirs.append(dirs[i])
        seen[vt] = len(loop_erased) - 1  # Actualizar índice más reciente
    return np.array(loop_erased), np.array(new_dirs)


"""import numpy as np


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("pop from empty stack")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            raise IndexError("peek from empty stack")

    def size(self):
        return len(self.items)


def dfsmatrix(M):
    indices = np.transpose(np.nonzero(1-M))
    num = np.random.randint(0, len(indices))
    i, j = indices[num]
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    curr = np.array([i, j])
    pila = Stack()
    out = []
    pila.push(curr)
    while not pila.is_empty():
        curr = pila.pop()
        t_curr = tuple(curr)
        if t_curr not in out:
            out.append(t_curr)
        for d in directions:
            next_ = curr + d
            t_next = tuple(next_)
            if M[t_next] == 0 and t_next not in out:
                pila.push(next_)
    return np.array(out)
"""