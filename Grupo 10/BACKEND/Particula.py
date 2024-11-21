class Particula:

    """
    Clase que representa una partícula en el grafo.
    Atributos:
    - posicion (int): La posición actual de la partícula.
    - bloqueado (bool): Indica si la partícula está bloqueada o no (si puede avanzar o no).
    Métodos:
    - __init__(posicion: int, bloqueado: bool): Inicializa una nueva instancia de la clase Particula.
    - avanzar(): Avanza la partícula una posición hacia adelante.
    - __str__(): Devuelve una representación en cadena de la partícula como str.
    """

    def __init__(self, calle, posicion: int, bloqueado: bool = False, id: int=0) -> None:
        self.calle = calle
        self.posicion = posicion
        self.bloqueado = bloqueado
        self.id = id
        
    def avanzar(self) -> None:
        self.posicion += 1

    def __str__(self) -> str:
        return f'Posición: {self.posicion}, Bloqueado: {self.bloqueado}'
