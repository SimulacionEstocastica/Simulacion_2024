import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ust import erase_loops, Grafo


def test_loops(path):
    nodos, count = np.unique(path, return_counts=True, axis=0)
    if (count > 1).any():
        print(f"Contiene loops, {np.sum(count>1)}")
    else:
        print("Todo bien")


if __name__ == '__main__':
    np.random.seed(9)
    for _ in range(100):
        grafo = Grafo()
        visited, dirs = grafo.random_walk(np.random.randint(0, 10, 2))
        print(visited)
        print(dirs)
        print("----")
        lerw, dir = erase_loops(visited, dirs)
        print(lerw)
        print(dir)
        test_loops(visited)
        test_loops(lerw)
    grafo = Grafo()
    grafo.wilson()
    print(grafo.grid)
