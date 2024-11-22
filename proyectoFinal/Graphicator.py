import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import Normalize
import pulp
from matplotlib.lines import lineStyles


class Graphicator:

    mat_adj = np.array([])
    points = np.array([])
    points_x = np.array([])
    points_y = np.array([])

    def city_creator_random(self, n, l):
        points = np.random.uniform(0,l,2*n)
        pointss = np.array([(points[i],points[n+i]) for i in range(n)])
        self.points = pointss
        self.points_x = np.array([points[i] for i in range(n)])
        self.points_y = np.array([points[n+i] for i in range(n)])
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(points[i] - points[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        self.mat_adj = distance_matrix

    def city_creator(self, arr):
        n = len(arr)
        self.points = np.array(arr)
        self.points_x = np.array([self.points[i][0] for i in range(n)])
        self.points_y = np.array([self.points[i][1] for i in range(n)])
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(arr[i] - arr[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        self.mat_adj = distance_matrix

    def find_intersect_distance(self,tourfinalA,tourfinalB):
        distance_tot = []
        for i in range(len(tourfinalA)):
            tour_A = tourfinalA[i]
            tour_B = tourfinalB[i]

            paresA = []
            for j in range(len(tour_A)-1):
                paresA.append((tour_A[j],tour_A[j+1]))
            paresA.append((tour_A[-1],tour_A[0]))

            paresB = []
            for j in range(len(tour_B)-1):
                paresA.append((tour_B[j],tour_B[j+1]))
            paresA.append((tour_B[-1],tour_B[0]))

            inter = []
            for a in paresA:
                k = (a[0],a[1])
                l = (a[1],a[0])
                if k in paresB or l in paresA:
                    inter.append(a)

            distace = 0

            for edge in inter:
                distace += self.mat_adj[edge[0],edge[1]]

            distance_tot.append(distace)

        return distance_tot


    def plot_distances(self, distances, other_max):
        t = [i for i in range(len(distances))]
        max = [other_max for ti in t]

        plt.plot(t, max,label="distance PuLP solution",linestyle="--", color='grey')
        plt.plot(t, distances,label="distance ACS", color='red')
        plt.title('Best ant tour distance on each iteration')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.legend()
        plt.show()

    def plot_distances_N(self, distancesA,distancesB,distanceInter):
        t = [i for i in range(len(distancesA))]
        distance_sum = [distancesA[i]+distancesB[i]-distanceInter[i] for i in range(len(distancesA))]
        plt.plot(t, distancesA, label="distance A", linestyle="--", color='red')
        plt.plot(t, distancesB, label="distance B",linestyle="--", color='blue')
        plt.plot(t, distance_sum, label="distance SUM", color='orange')
        plt.plot(t, distanceInter, label="distance inter",linestyle="--", color='green')
        plt.title('Best ant tour distance on each iteration')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.legend()
        plt.show()

    def plot_all(self, ants):
        tours = [ant.tour for ant in ants]
        n = len(self.points)
        plt.figure(figsize=(8, 8))
        plt.grid(False)

        points_x = np.array([point[0] for point in self.points])
        points_y = np.array([point[1] for point in self.points])

        # Plot edges
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(self.points[i] - self.points[j]) > 0:
                    plt.plot([self.points[i][0], self.points[j][0]], [self.points[i][1], self.points[j][1]], color='gray', linewidth=2.5)

        # Highlight the tour
        for tour in tours:
            for i in range(n):
                plt.plot([self.points[tour[i]][0], self.points[tour[i + 1]][0]], [self.points[tour[i]][1], self.points[tour[i + 1]][1]], color='red', linewidth=2)
            #plt.plot([self.points[tour[-1]][0], self.points[tour[0]][0]], [self.points[tour[-1]][1], self.points[tour[0]][1]], color='red', linewidth=2)

        # Plot points
        plt.scatter(points_x, points_y, c='blue', s=300)

        plt.title('Complete Graph with Tour')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def pulp_solution(self):
        
        n = len(self.mat_adj)
        
        # Definir el modelo de optimización
        model = pulp.LpProblem("TSP", pulp.LpMinimize)
        
        # Variables de decisión: x[i, j] indica si se viaja de i a j
        x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(n) if i != j], cat="Binary")
        # Variables para la restricción de subtour
        u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n-1, cat="Continuous")
        
        # Función objetivo: minimizar la distancia total
        model += pulp.lpSum(self.mat_adj[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
        
        # Restricción: debe salir exactamente un arco de cada nodo
        for i in range(n):
            model += pulp.lpSum(x[i, j] for j in range(n) if i != j) == 1
        
        # Restricción: debe entrar exactamente un arco en cada nodo
        for j in range(n):
            model += pulp.lpSum(x[i, j] for i in range(n) if i != j) == 1
        
        # Restricciones de subtour para evitar ciclos
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model += u[i] - u[j] + n * x[i, j] <= n - 1
        
        # Resolver el modelo
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Obtener la solución óptima
        pre_tour = []
        if model.status == pulp.LpStatusOptimal:
            pre_tour = [(i, j) for i in range(n) for j in range(n) if i != j and pulp.value(x[i, j]) == 1]
        tour = [0]
        while len(tour) < n + 2:
            i = tour[-1]
            for edge in pre_tour:
                if edge[0] == i:
                    tour.append(edge[1])

        # Graficar la solución
        fig, ax = plt.subplots()
        
        points_x = np.array([point[0] for point in self.points])
        points_y = np.array([point[1] for point in self.points])
        ax.scatter(points_x, points_y, c='blue', s=300, label="Ciudades")

        for i, txt in enumerate(range(n)):
            ax.annotate(txt, (points_x[i], points_y[i]), textcoords="offset points", xytext=(5,5), ha='center')

        # Dibujar el tour
        tour_x = [self.points[tour[i % n]][0] for i in range(n + 1)]
        tour_y = [self.points[tour[i % n]][1] for i in range(n + 1)]
        
        ax.plot(tour_x, tour_y, color='red', linewidth=2, label="Recorrido óptimo")
        
        ax.set_title('Recorrido óptimo del viajante')
        ax.legend()
        plt.show()

        return tour, pulp.value(model.objective)
    

    def plot_final_pheromones(self, history, edges=True, title='Pheromones trail on iteration '):
        n = len(self.points)
        m = len(history)
        ##G = nx.complete_graph(n)
        # pos = {i: (point[0], point[1]) for i, point in enumerate(self.points)}
        fig, ax = plt.subplots()

        points_x = np.array([point[0] for point in self.points])
        points_y = np.array([point[1] for point in self.points])
        ciudades = ax.scatter(points_x, points_y, c='blue', s=300)

        stacked_matrices = np.array(history)

        # Obtenemos el valor máximo
        max_value = np.max(stacked_matrices)

        colormap = cm.get_cmap('Greens')

        norm = Normalize(vmin=0, vmax=1)  # Normaliza los valores de feromonas al rango [0, 1]

        # Añadir la barra de colores al gráfico
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Necesario para la barra de colores
        fig.colorbar(sm, ax=ax, label='Pheromones (Normalized)')  # Mostrar barra de colores con etiqueta

        L = []

        for frame in range(m):
            # ciudades = nx.draw_networkx_nodes(G, pos, node_color='blue', ax=ax)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if history[frame][i][j] > 0:
                        intensity = history[frame][i][j] / max_value  # Evitar división por cero
                        color = colormap(intensity)  # Obtener el color del colormap
                        # edge = nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], width=2, alpha=0.6, edge_color=plt.cm.viridis(intensity), ax=ax)
                        edge = ax.plot([self.points[i][0], self.points[j][0]], [self.points[i][1], self.points[j][1]],
                                       color=color, linewidth=2 * intensity + 0.75)
                        edges.extend(edge)

            title = ax.text(0.5,1.05, 'Pheromones trail on iteration '+str(frame), ha='center', va='center', transform=ax.transAxes)
            L.append([ciudades] + edges+[title])
        ani = animation.ArtistAnimation(fig, L, interval=250, repeat=True)
        #
        writer = animation.PillowWriter(fps=5,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('tmp/pheromone_animation.gif', writer=writer)
        plt.show()

        def update(frame):
            """
            for line in ax.get_lines():  # Eliminar líneas existentes
                line.remove()
            for i in range(n):
                for j in range(i + 1, n):
                    pheromone_value = history[frame][i][j]
                    intensity = pheromone_value / (max_value + 1e-5)  # Evitar división por cero
                    color = colormap(intensity)  # Obtener el color del colormap
                    L = ax.plot([self.points[i][0], self.points[j][0]],
                                [self.points[i][1], self.points[j][1]],
                                color=color, linewidth=2 * intensity + 0.75)
            return ax.get_lines()
            """
            return L[frame]

        ani = animation.FuncAnimation(fig=fig, func=update, frames=m, interval=300, blit=True)

        plt.show()


    def plot_points(self):
        plt.scatter(self.points_x, self.points_y,c='blue',s=300)
        n = len(self.points)
        arcos_x = []
        arcos_y = []
        for i in range(n):
            for j in range(i + 1, n):
                arcos_x.append([self.points[i][0], self.points[j][0]])
                arcos_y.append([self.points[i][1], self.points[j][1]])
                plt.plot(arcos_x[-1], arcos_y[-1], linestyle='dashed',color='gray', linewidth=0.75)
        plt.title('Configuration of cities')
        plt.show()

    def plot_tour(self,ant,edges=True):
        n = len(self.points)
        tour = ant.tour
        #plt.figure(figsize=(8, 8))

        fig, ax = plt.subplots()

        points_x = np.array([point[0] for point in self.points])
        points_y = np.array([point[1] for point in self.points])
        ciudades = ax.scatter(points_x, points_y, c='blue', s=300)

        if edges:
            arcos_x = []
            arcos_y = []
            for i in range(n):
                for j in range(i + 1, n):
                    arcos_x.append([self.points[i][0], self.points[j][0]])
                    arcos_y.append([self.points[i][1], self.points[j][1]])
                    ax.plot(arcos_x[-1], arcos_y[-1], linestyle='dashed',color='gray', linewidth=0.75)

        tour_x = []
        tour_y = []
        for i in range(n):
            tour_x.append([self.points[tour[i]][0],self.points[tour[i+1]][0]])
            tour_y.append([self.points[tour[i]][1], self.points[tour[i+1]][1]])

            # plt.plot([self.points[tour[-1]][0], self.points[tour[0]][0]], [self.points[tour[-1]][1], self.points[tour[0]][1]], color='red', linewidth=2)

        tour_plot =  ax.plot(tour_x[0], tour_y[0], color='red')[0]
        ax.set_title('Best ant path found')

        def update(frame):
            # for each frame, update the data stored on each artist.
            x = tour_x[:frame]
            y = tour_y[:frame]
            # update the line plot:
            tour_plot.set_xdata(tour_x[:frame])
            tour_plot.set_ydata(tour_y[:frame])
            return (ciudades,tour_plot)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=n+1, interval=250)
        writer = animation.PillowWriter(fps=5,
                                         metadata=dict(artist='Me'),
                                         bitrate=1800)
        ani.save('tmp/best_tour_for_tsp.gif', writer=writer)
        plt.show()

    def plot_tours(self,ants,edges=True):
        n = len(self.points)
        #tour = ant.tour
        #plt.figure(figsize=(8, 8))

        fig, ax = plt.subplots()

        points_x = np.array([point[0] for point in self.points])
        points_y = np.array([point[1] for point in self.points])
        ciudades = ax.scatter(points_x, points_y, c='blue', s=300)


        if edges:
            arcos_x = []
            arcos_y = []
            for i in range(n):
                for j in range(i + 1, n):
                    arcos_x.append([self.points[i][0], self.points[j][0]])
                    arcos_y.append([self.points[i][1], self.points[j][1]])
                    ax.plot(arcos_x[-1], arcos_y[-1], linestyle='dashed',color='gray', linewidth=0.75)
        tour_fin_x = []
        tour_fin_y = []
        for i in range(len(ants)):
            tour = ants[i]
            tour_x = []
            tour_y = []
            for i in range(n):
                tour_x.append([self.points[tour[i]][0], self.points[tour[i + 1]][0]])
                tour_y.append([self.points[tour[i]][1], self.points[tour[i + 1]][1]])
            tour_fin_x.append(tour_x)
            tour_fin_y.append(tour_y)

            # plt.plot([self.points[tour[-1]][0], self.points[tour[0]][0]], [self.points[tour[-1]][1], self.points[tour[0]][1]], color='red', linewidth=2)

        tour_plot =  ax.plot(tour_x[0], tour_y[0], color='red')[0]

        def update(frame):
            # for each frame, update the data stored on each artist.
            x = tour_fin_x[frame]
            y = tour_fin_y[frame]
            # update the line plot:
            tour_plot.set_xdata(tour_fin_x[frame])
            tour_plot.set_ydata(tour_fin_y[frame])
            ax.set_title('Path best ant iteration '+str(frame))
            #ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')
            #tour_plot.set_label('Iteracion:'+str(frame))
            #tour_plot.legend()
            return (ciudades,tour_plot)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(ants), interval=250)
        writer = animation.PillowWriter(fps=5,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('tmp/best_tours_animation_for_tsp.gif', writer=writer)
        plt.show()

    def plot_final_tours(self,antA, antB, edges=False):
        n = len(self.points)

        tourA = antA.tour
        tourB = antB.tour

        #plt.figure(figsize=(8, 8))

        fig, ax = plt.subplots()

        points_x = np.array([point[0] for point in self.points])
        points_y = np.array([point[1] for point in self.points])
        ciudades = ax.scatter(points_x, points_y, c='blue', s=300)

        if edges:
            arcos_x = []
            arcos_y = []
            for i in range(n):
                for j in range(i + 1, n):
                    arcos_x.append([self.points[i][0], self.points[j][0]])
                    arcos_y.append([self.points[i][1], self.points[j][1]])
                    ax.plot(arcos_x[-1], arcos_y[-1], linestyle='dashed',color='gray', linewidth=0.75)

        tour_xA = []
        tour_yA = []
        tour_xB = []
        tour_yB = []
        for i in range(len(tourA)-1):
            tour_xA.append([self.points[tourA[i]][0],self.points[tourA[i+1]][0]])
            tour_yA.append([self.points[tourA[i]][1], self.points[tourA[i+1]][1]])

        for i in range(len(tourB)-1) :   
            tour_xB.append([self.points[tourB[i]][0],self.points[tourB[i+1]][0]])
            tour_yB.append([self.points[tourB[i]][1], self.points[tourB[i+1]][1]])
            

            # plt.plot([self.points[tour[-1]][0], self.points[tour[0]][0]], [self.points[tour[-1]][1], self.points[tour[0]][1]], color='red', linewidth=2)

        tour_plotA =  ax.plot(tour_xA[0], tour_yA[0], color='red', linestyle='--', alpha=0.5)[0]
        tour_plotB =  ax.plot(tour_xB[0], tour_yB[0], color='blue', linestyle='--', alpha=0.5)[0]
        ax.set_title('Best ant path for last iteration')

        def update(frame):
            # update the line plot:
            tour_plotA.set_xdata(tour_xA[:frame])
            tour_plotA.set_ydata(tour_yA[:frame])
            tour_plotB.set_xdata(tour_xB[:frame])
            tour_plotB.set_ydata(tour_yB[:frame])
            return (ciudades,tour_plotA, tour_plotB)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=n+1, interval=250)
        writer = animation.PillowWriter(fps=5,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('tmp/best_tour_animation_for_2tsp.gif', writer=writer)
        plt.show()

    def plot_final_all_torus(self,antsA,antsB,edges=False):
        n = len(self.points)
        #tour = ant.tour
        #plt.figure(figsize=(8, 8))

        fig, ax = plt.subplots()

        points_x = np.array([point[0] for point in self.points])
        points_y = np.array([point[1] for point in self.points])
        ciudades = ax.scatter(points_x, points_y, c='blue', s=300)


        if edges:
            arcos_x = []
            arcos_y = []
            for i in range(n):
                for j in range(i + 1, n):
                    arcos_x.append([self.points[i][0], self.points[j][0]])
                    arcos_y.append([self.points[i][1], self.points[j][1]])
                    ax.plot(arcos_x[-1], arcos_y[-1], linestyle='dashed',color='gray', linewidth=0.75)
        tour_fin_xA = []
        tour_fin_yA = []
        tour_fin_xB = []
        tour_fin_yB = []
        for i in range(len(antsA)):
            tourA = antsA[i].tour
            tourB = antsB[i].tour
            tour_xA = []
            tour_yA = []
            tour_xB = []
            tour_yB = []
            for i in range(len(tourA)-1):
                tour_xA.append([self.points[tourA[i]][0], self.points[tourA[i + 1]][0]])
                tour_yA.append([self.points[tourA[i]][1], self.points[tourA[i + 1]][1]])
            for i in range(len(tourB)-1):
                tour_xB.append([self.points[tourB[i]][0], self.points[tourB[i + 1]][0]])
                tour_yB.append([self.points[tourB[i]][1], self.points[tourB[i + 1]][1]])
            tour_fin_xA.append(tour_xA)
            tour_fin_yA.append(tour_yA)
            tour_fin_xB.append(tour_xB)
            tour_fin_yB.append(tour_yB)

            # plt.plot([self.points[tour[-1]][0], self.points[tour[0]][0]], [self.points[tour[-1]][1], self.points[tour[0]][1]], color='red', linewidth=2)

        tour_plotA =  ax.plot(tour_xA[0], tour_yA[0], color='red', linestyle='--', alpha=0.5)[0]
        tour_plotB = ax.plot(tour_xB[0], tour_yB[0], color='blue', linestyle='--', alpha=0.5)[0]

        def update(frame):
            # update the line plot:
            tour_plotA.set_xdata(tour_fin_xA[frame])
            tour_plotA.set_ydata(tour_fin_yA[frame])
            tour_plotB.set_xdata(tour_fin_xB[frame])
            tour_plotB.set_ydata(tour_fin_yB[frame])
            ax.set_title('Path best ant iteration '+str(frame))
            #ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')
            #tour_plot.set_label('Iteracion:'+str(frame))
            #tour_plot.legend()
            return (ciudades,tour_plotA, tour_plotB)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(antsA), interval=250)
        writer = animation.PillowWriter(fps=5,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('tmp/best_tours_animation_for_2tsp.gif', writer=writer)
        plt.show()