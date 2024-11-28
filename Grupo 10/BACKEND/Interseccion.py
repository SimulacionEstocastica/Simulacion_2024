class Interseccion():

    """
    Clase que representa una intersección en un sistema de tráfico.
    Atributos:
    posicion : tupla, coordenadas de la intersección.
    bloqueado : list, indica si la intersección está bloqueada en ciertas direcciones.
    calles : tuple, contiene las calles conectadas a la intersección.

    Métodos:
    __str__() -> str: Devuelve una representación en cadena de la intersección.
    update_bloqueo(direccion): Actualiza el estado de bloqueo de la calle en la dirección especificada.
    """
        
    def __init__(self, posicion: tuple, bloqueado: list, calles: tuple) -> None:
        self.posicion = posicion
        self.bloqueado = bloqueado
        self.calles = calles

    def __str__(self) -> str:
        return f'Posición: {self.posicion}, Bloqueado: {self.bloqueado}'

    def update_bloqueo(self, direccion):
        self.calles[direccion].update_bloqueo()