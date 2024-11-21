# City TASEP

Este proyecto cuenta con licencia conforme a los términos de la licencia MIT

___
## Como ejecutar el sandbox:

Para ejecutar el sandbox se puede hacer de la siguiente manera:

Instalación de dependencias:

0. Descargar Node.js de la página oficial: https://nodejs.org/es/download/ y clonar el repositorio.

1. Crear un entorno virtual de python usando conda o virtualenv e instalar las dependencias del archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

2. Instalar las dependencias de Node.js: Abrir una terminal en la carpeta `/PAGINA_TASEP` y ejecutar el siguiente comando:

```bash
npm install
```

Ejecución:

3. Abrir una terminal en la carpeta `/PAGINA_TASEP` y ejecutar el siguiente comando:

```bash
npm run dev
```

y abrir en el navegador la dirección que se muestra en la terminal. (Para detener el servidor, presionar `Ctrl + C` en la terminal)

4. Abrir una termina en la carpeta `/BACKEND`, activar el entorno virtual creado anteriormente y ejecutar el siguiente comando:

```bash
python app.py
```

5. Disfrutar de las simulaciones.
____

## Resumen del proyecto:

## Introducción

El *Totally Asymmetric Simple Exclusion Process* (TASEP) es un proceso estocástico que modela sistemas de partículas que se mueven unidireccionalmente bajo reglas de exclusión. Matemáticamente, se describe como una cadena de Markov a tiempo discreto con estados en $\{0,1\}^\mathbb{Z}$, donde el valor $1$ en una posición indica la presencia de una partícula, y $0$ su ausencia. Un paso de la cadena es una actualización: las partículas avanzan a la posición siguiente con una cierta probabilidad, siempre que esté vacía. Se puede interpretar como flujo de vehículos en una calle unidireccional.

En este proyecto, se estudia el comportamiento de múltiples calles que se intersectan en un plano bidimensional, con el objetivo de analizar sus propiedades y características fundamentales.

---

## Desarrollo

El TASEP puede actualizarse mediante dos métodos principales: **secuencial** y **paralelo**.

**Actualización secuencial:**
Las partículas se actualizan sucesivamente en orden decreciente según su posición. Aquí dos partículas consecutivas pueden avanzar juntas en una misma actualización. En sistemas con múltiples calles, las calles se actualizan siguiendo un orden predeterminado.

**Actualización paralela:** Todas las partículas intentan avanzar simultáneamente. Si una partícula está bloqueada por otra, no podrá moverse, incluso si la partícula bloqueadora avanza. En sistemas con varias calles, se selecciona una permutación uniforme del orden de las calles para determinar la secuencia de actualización.

Para modelar el sistema, se implementó un conjunto de clases que representan partículas, calles e intersecciones, incorporando todos los parámetros relevantes del TASEP. Las actualizaciones se simulan mediante iteraciones sobre las partículas, siguiendo las reglas del método de actualización (secuencial o paralelo). En el caso de intersecciones, se bloquea el avance de una partícula si intenta moverse a una posición ocupada por otra partícula proveniente de otra calle.

Un aspecto clave del análisis es el *proceso de crecimiento*, definido como una función de altura $h(t)$, con valores en $\mathbb{Z}^\mathbb{Z}$. La altura se inicializa como:

$$
h(0, i) =
\begin{cases}
0 & \text{si } i = 0, \\
h(0, i-1) - 1 & \text{si } i > 0, \\
h(0, i+1) + 1 & \text{si } i < 0.
\end{cases}
$$

En cada paso, la altura se incrementa en $1$ en las posiciones donde una partícula avanza, manteniéndose constante en las posiciones donde no ocurre movimiento.

Se analizaron dos configuraciones iniciales relevantes:

- Partículas ocupando todas las posiciones hacia atrás desde un punto dado y vacías hacia adelante. Para el caso de una calle, $h(t)$ tiende a una parábola.
- Partículas intercaladas en $\mathbb{Z}$. Para el caso de una calle, $h(t)$ tiende a una recta.

Ambos casos fueron simulados en un plano bidimensional con intersecciones, permitiendo estudiar cómo estas perturban las convergencias conocidas.

- **Tiempo de cruce**: Variable aleatoria que representa el número de iteraciones necesarias para que una partícula supere una intersección.
- **Tasa de cruce**: Variable aleatoria que representa la cantidad de partículas que superan una intersección por unidad de tiempo.

Para ambas variables se estimaron sus valores esperados mediante simulaciones Monte Carlo, analizando su dependencia con la probabilidad de avance de las partículas en cada calle.

---

### Resultados

Las simulaciones permitieron modelar el TASEP para diferentes configuraciones y parámetros, extendiéndolo a sistemas bidimensionales con múltiples intersecciones. Se identificaron perturbaciones en las convergencias del proceso de crecimiento en ambas configuraciones iniciales, y se observaron relaciones entre las probabilidades de avance de una calle en función de la velocidad de la otra en puntos de intersección.

Además, se obtuvieron estimaciones para los valores esperados del tiempo de cruce y la tasa de cruce, revelando patrones que sugieren posibles optimizaciones en sistemas de tráfico unidireccional.

___


