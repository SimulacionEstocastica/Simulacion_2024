import matplotlib.pyplot as plt
import random
from Interseccion import Interseccion
class Calles():

    def __init__(self, calles: list) -> None:
        self.calles = calles
        self.calles_x = [c for c in calles if c.direccion == 0]
        self.calles_y = [c for c in calles if c.direccion == 1]
        self.intersecciones = []

    def update_intersecciones(self) -> None:
        for calle_x in self.calles_x:
            for calle_y in self.calles_y:
                choque_x = False
                choque_y = False
                for particula in calle_x:
                    if particula.posicion == calle_y.posicion:
                        choque_x = True
                        break
                for particula in calle_y:
                    if particula.posicion == calle_x.posicion:
                        choque_y = True
                        break
                if choque_x and choque_y:
                    raise ValueError('Hay un choque')         
                interseccion = Interseccion((calle_x.posicion, calle_y.posicion), [choque_x, choque_y], (calle_x, calle_y))
                calle_x.agregar_interseccion(interseccion)
                calle_y.agregar_interseccion(interseccion)
                self.intersecciones.append(interseccion)

    def update_bloqueos(self) -> None:
        for calle in self.calles:
            calle.update_bloqueo()

    def update_estado_intersecciones(self) -> None:
        for calle in self.calles:
            calle.update_estado_intersecciones()

    def update_secuencial(self, p: float) -> None:
        for calle in self.calles:
            # Actualiza una calle
            calle.update_secuencial(p)
            # Después de actualizar una calle es necesario actualizar los bloqueos de todas las calles
            self.update_bloqueos()

    def update_paralelo(self, p: float) -> None:
        # Definir un orden para iterar (de manera uniforme para simular paralelismo)
        calles_copy = self.calles.copy()
        random.shuffle(calles_copy)
        # for bajo ese orden
        for calle in calles_copy:
            # Actualiza una calle
            calle.update_paralelo(p)
        # Después de actualizar todas las calles es necesario actualizar los bloqueos de todas las calles
            self.update_bloqueos()

    def plot(self) -> None:
        for calle in self.calles:
            calle.plot()

    def plot_2d(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.grid(True)  # Agregar grilla
        
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']  # Lista de colores para las calles
        color_index = 0
        
        for interseccion in self.intersecciones:
            ax.plot(interseccion.posicion[1], interseccion.posicion[0], 'ko')
        
        for calle in self.calles_x:
            color = colors[color_index % len(colors)]
            ax.plot([p.posicion for p in calle], [calle.posicion for _ in calle], color + 'o', label=f'Calle X {calle.posicion}')
            color_index += 1
        
        for calle in self.calles_y:
            color = colors[color_index % len(colors)]
            ax.plot([calle.posicion for _ in calle], [p.posicion for p in calle], color + 'o', label=f'Calle Y {calle.posicion}')
            color_index += 1
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    def delete_particulas_posicion(self, posicion: int) -> None:
        for calle in self.calles:
            calle.delete_particulas_posicion(posicion)

    def add_calle(self, calle) -> None:
        self.calles.append(calle)
        if calle.direccion == 0:
            self.calles_x.append(calle)
        else:
            self.calles_y.append(calle)

    def agregar_particulas_inicio(self, initial_id: int = 0, p: float = 1) -> list:
        particulas_agregadas = []
        for calle in self.calles:
            nueva_particula = calle.agregar_particula_inicio(initial_id, p)
            if nueva_particula is not None:
                particulas_agregadas.append(nueva_particula)
                initial_id += 1
        return particulas_agregadas

    def vaciar_objeto(self) -> None:
        self.calles = []
        self.calles_x = []
        self.calles_y = []
        self.intersecciones = []

    def get_particulas(self) -> list:
        particulas = []
        for calle in self.calles:
            particulas += calle
        return particulas

