import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Particula import Particula
from Interseccion import Interseccion



class Calle(list):

    """
    Clase que representa una calle en un modelo de simulación TASEP.
    Métodos:
    - __init__(*args): Constructor de la clase.
    - agregar_particula_inicio(): Agrega una partícula al inicio de la calle.
    - update_bloqueo_casilla(i: int): Actualiza el estado de bloqueo de una casilla en la calle.
    - update_bloqueo(): Actualiza el estado de bloqueo de todas las casillas en la calle.
    - update_secuencial(p: float): Actualiza la posición de las partículas en la calle de forma secuencial.
    - update_paralelo(p: float): Actualiza la posición de las partículas en la calle de forma paralela.
    - plot(): Grafica la posición de las partículas en la calle.
    - animate(n: int, p: float, mode: str): Genera una animación de la simulación.
    - __str__(): Retorna una representación en cadena de la calle.
    """

    def __init__(self, *args, direccion: int, intersecciones: list, posicion: int) -> None:
        super().__init__(*args)
        self.direccion = direccion
        self.intersecciones = intersecciones
        self.posicion = posicion
        self.altura = None
        self.inv = None

    
    def inversa(self,u):
            for i in range(1, len(self)+1):
                if self[-i].posicion<=u:
                    return i-1


    def iniciar_altura(self, Min, Max) -> None:
        inv=self.inversa(-1)
        dict={}
        
        for u in range(Min, Max+1):
            try:
                dict[u]=2*(self.inversa(u-1)-inv)+u
            except:
                dict[u]=0
        self.altura=dict
        self.inv=inv
    
    def actualizar_altura(self) -> None:
        Min=list(self.altura.keys())[0]
        Max=list(self.altura.keys())[-1]
        dict={}

        for u in range(Min, Max+1):
            try:
                dict[u]=2*(self.inversa(u-1)-self.inv)+u
            except:
                dict[u]=-1
        self.altura=dict              


    def agregar_particula_inicio(self, id: int = 0, p: float = 1) -> None:
        nueva_particula = Particula(self, 0, False, id)
        if len(self) == 0:
            self.append(nueva_particula)
            self.update_estado_intersecciones()
            return nueva_particula
        elif self[0].posicion != 0 and np.random.rand() < p:
            self.insert(0, nueva_particula)
            self.update_bloqueo()
            self.update_estado_intersecciones()
            return nueva_particula

    def agregar_interseccion(self, interseccion: Interseccion) -> None:
        self.intersecciones.append(interseccion)

    def delete_particulas_posicion(self, posicion: int) -> None:
        for i, p in enumerate(self):
            if p.posicion > posicion:
                break

    def update_estado_intersecciones(self) -> None:
        for interseccion in self.intersecciones:
            bloqueado = False
            for particula in self:
                if particula.posicion == interseccion.posicion[(self.direccion + 1) % 2]:
                    bloqueado = True
                    # La particula esta en la interseccion, bloquea el paso en la otra direccion
                    interseccion.bloqueado[(self.direccion + 1) % 2] = True
                    break
            # Ninguna particula de esta direccion esta en la interseccion, desbloquea el paso en la otra direccion
            if not bloqueado:
                interseccion.bloqueado[(self.direccion + 1) % 2] = False
            

    def update_bloqueo_casilla(self, i: int) -> None:
        # La de mas adelante no esta bloqueada (excepto por el if de abajo)
        if i == len(self) - 1:
            self[i].bloqueado = False
        else:
            self[i].bloqueado = self[i+1].posicion == self[i].posicion + 1
        # Si la particula no esta bloqueada en su calle, revisar si la interseccion la bloquea
        if not self[i].bloqueado:
            for interseccion in self.intersecciones:
                # Si la siguiente posicion de la particula es la posicion de la interseccion en la direccion de la calle
                # entonces la interseccion bloquea la particula
                if (self[i].posicion +1) == interseccion.posicion[(self.direccion + 1) % 2]:
                    self[i].bloqueado = interseccion.bloqueado[self.direccion]
                    break
    
    
    # Buena
    def update_secuencial(self, p: float) -> None:
        for i in range(len(self)-1, -1, -1):
            if not self[i].bloqueado and np.random.rand() < p:
                self[i].avanzar()
                # Verificar si se bloquea la casilla luego de avanzar
                self.update_bloqueo_casilla(i)
                # Verificar si se desbloquea la casilla anterior luego de avanzar (si no es la última casilla)
                if i != 0:
                    self.update_bloqueo_casilla(i-1)
        # Actualizar el estado de las intersecciones para las que se usen en las siguientes calles
        self.update_estado_intersecciones()
    
    def update_bloqueo(self) -> None:
        for i in range(len(self)):
            self.update_bloqueo_casilla(i)    

    def update_paralelo(self, p: float) -> None:
        for i in range(len(self)-1, -1, -1):
            if not self[i].bloqueado:
                if np.random.rand() < p:
                    self[i].avanzar()
        # Los bloqueos se actualizan al final de la iteracion
        self.update_bloqueo()
        # Actualizar el estado de las intersecciones para las que se usen en las siguientes calles
        self.update_estado_intersecciones()

    def plot(self) -> None:
        plt.plot([p.posicion for p in self], [0 for _ in self], 'ro')
        plt.show()

    def animate(self, n: int, p: float, mode: str) -> None:
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'ro')

        def init():
            ax.set_xlim(0, len(self))
            ax.set_ylim(0, 1)
            return line,

        def update(frame):
            if mode == 'secuencial':
                self.update_secuencial(p)
            elif mode == 'paralelo':
                self.update_paralelo(p)
            line.set_data([p.posicion for p in self], [0 for _ in self])
            ax.set_xlim(0, len(self))
            return line,

        ani = animation.FuncAnimation(fig, update, frames=n, init_func=init, blit=True)
        ani.save('animation.gif', writer='pillow')

    def __str__(self) -> str:
        return '\n'.join([str(p) for p in self])
