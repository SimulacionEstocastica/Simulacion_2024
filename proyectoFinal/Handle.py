def caso_inicial():
    print("Bienvenido a la simulación! \n")
    choice = False
    while not choice:
        choice2 = int(input("elija con 1 si desea construir la matriz \n elija con 2 si desea tippear la matriz \n"))
        print("elegiste ", choice2, ". Es esto correcto?")
        b = int(input("1 para si, cualquier otra tecla para no"))
        if b == 1:
            if choice2 == 1:
               choice = True
            elif choice2 == 2:
             choice = True
            else:
                print("esa no es opcion!")
    return choice2


class Handle:
    N = 0
    def __init__(self):
        eleccion = caso_inicial()

    def construir_matriz(self):
        print("elegiste construir matriz \n")
        choice = False
        while not choice:
            a = int(input("Cuantas ciudades tendrás? (entre 1 y 50)"))
            print("elegiste", a, "ciudades. Es esto correcto?")
            b = int(input("1 para si, cualquier otra tecla para no"))

        self.N = a


