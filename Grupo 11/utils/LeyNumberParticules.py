import numpy as np

class Ley_Number_Particules:
    def __init__(self, ley, ley_growth, **kwargs) -> None:
        self.ley_name = ley
        self.kwargs = kwargs["kwargs"]
        self.ley_growth = ley_growth
        self.call_func = self._choose_func()

    def _growth_ley(self, N_init, **kwargs):
        
        match self.ley_growth:
            case "stable":
                return 0
            case "Bernoulli":
                return int(np.random.random() < self.kwargs['p'])
            case "Binomial":
                return (np.random.binomial(N_init, self.kwargs['p']))

    def _spwan_left_func(self, positions, dist, case , N, spawn_rate = 1.005):
        pass

    def _spawn_between_func(self, positions, dist, case, N):
        if case == "uni":
            positions = positions.astype(int)
            N_spawn = self._growth_ley(len(positions))
            interval = np.arange(positions[0], positions[-1]+1)
            mask = np.ones(interval.shape[0], dtype=np.bool_)
            mask[positions-positions[0]] = 0
            new_pos = np.random.choice(interval[mask], N_spawn)
            positions = np.sort(np.concatenate((positions,new_pos)))
            dist = positions[1:] - positions[:-1]
            return positions.shape[0],positions, dist, new_pos

        elif case == "multi":
            N_spawn = self._growth_ley(positions[0].shape[0]+positions[1].shape[0], **self.kwargs)
            if N_spawn == 0:
                return N, positions, dist, 0
            else:
                positions[0], positions[1] = positions[0].astype(int), positions[1].astype(int)
            N_spawn_x = np.random.binomial(N_spawn, 0.5)
            N_spawn_y = N_spawn - N_spawn_x

            interval = np.arange(positions[0][0], positions[0][-1]+1)
            mask = np.ones(interval.shape[0], dtype=np.bool_)
            mask[positions[0]-positions[0][0]] = 0
            
            new_pos = np.random.choice(interval[mask], N_spawn_x, replace=False)
            positions[0] = np.sort(np.concatenate((positions[0],new_pos)))

            interval = np.arange(positions[1][0], positions[1][-1]+1)
            mask = np.ones(interval.shape[0], dtype=np.bool_)
            mask[positions[1]-positions[1][0]] = 0
            new_pos = np.random.choice(interval[mask], N_spawn_y, replace=False)
            positions[1] = np.sort(np.concatenate((positions[1],new_pos)))

            x = self.kwargs["x_axis"]
            y = self.kwargs["y_axis"]

            if x in positions[1] and y in positions[0]:
                choice = np.random.random() < 0.5
                if choice == 0:
                    
                    interval = np.arange(positions[0][0], positions[0][-1]+1)
                    mask = np.ones(interval.shape[0], dtype=np.bool_)
                    mask[positions[0]-positions[0][0]] = 0
                    new_pos = np.random.choice(interval[mask], size = 1)
                    positions[0] = np.sort(np.concatenate((positions[0],new_pos)))

                    mask = np.ones(positions[0].shape[0]).astype(np.bool_)
                    ind_x = np.searchsorted(positions[0], y)
                    mask[ind_x] = False
                    positions[0] = positions[0][mask]

                elif choice == 1:

                    interval = np.arange(positions[1][0], positions[1][-1]+1)
                    mask = np.ones(interval.shape[0], dtype=np.bool_)
                    mask[positions[1]-positions[1][0]] = 0
                    new_pos = np.random.choice(interval[mask], size = 1)
                    positions[1] = np.sort(np.concatenate((positions[1],new_pos)))

                    mask = np.ones(positions[1].shape[0]).astype(np.bool_)
                    ind_x = np.searchsorted(positions[1], x)
                    mask[ind_x] = False
                    positions[1] = positions[1][mask]


            dist[0] = positions[0][1:] - positions[0][:-1]
            dist[1] = positions[1][1:] - positions[1][:-1]
            
            return positions[0].shape[0] + positions[1].shape[0],positions, dist, new_pos
        
        elif case == "multi_HD":
            x = self.kwargs["x_axis"]
            y = self.kwargs["y_axis"]

            n_axis = x.shape[0] + y.shape[0]

            N_spawn = self._growth_ley(N, **self.kwargs)
            rnd_axis = np.random.choice(n_axis, N_spawn)
            new_pos = 0
            for k in range(N_spawn):
                id_1 = rnd_axis[k] >= x.shape[0]
                id_2 = rnd_axis[k] % x.shape[0]
                interval = np.arange(positions[id_1][id_2][0], positions[id_1][id_2][-1]+1)
                mask = np.ones(interval.shape[0], dtype=np.bool_)
                positions[id_1][id_2] = positions[id_1][id_2].astype(int)
                mask[positions[id_1][id_2] - positions[id_1][id_2][0]] = False

                new_pos = np.random.choice(interval[mask], size = 1)
                positions[id_1][id_2] = np.sort(np.concatenate((positions[id_1][id_2],new_pos)))
                #May have some bugs ocasionnaly (if a particle spawns at an axis crossing)

                dist[id_1][id_2] = positions[id_1][id_2][1:] - positions[id_1][id_2][:-1]

            return N+N_spawn, positions, dist, new_pos
            

    def _spawn_fixed(self, positions, dist, case, N):
        if case == "uni":
            spawn_position = self.kwargs["spawn_pos"]
            N_spawn = self._growth_ley(N, **self.kwargs)
            if spawn_position != None and spawn_position not in positions and N_spawn > 0:
                positions = np.sort(np.concatenate((positions,[spawn_position]))).astype(int)
                dist = positions[1:] - positions[:-1]
                N += 1
            return N, positions, dist, [spawn_position]
        elif case == "multi":
            spawn_positions = self.kwargs["spawn_pos"]
            N_spawn = self._growth_ley(N, **self.kwargs)
            match N_spawn:
                case 0:
                    return N, positions, dist
                case 1:
                    ax = int(np.random.random() < 0.5)
                    if spawn_positions[ax] != None and spawn_positions[ax] not in positions[ax]:
                        positions[ax] = np.sort(np.concatenate((positions[ax],[spawn_positions[ax]])))
                        N += 1
                        dist[ax] = positions[ax][1:] - positions[ax][:-1]
                    return N, positions, dist, [spawn_position]
                case 2:
                    if spawn_positions[0] != None and spawn_positions[0] not in positions[0]:
                        positions[0] = np.sort(np.concatenate((positions[0],[spawn_positions[0]])))
                        N += 1
                        dist[0] = positions[0][1:] - positions[0][:-1]
                    if spawn_positions[1] != None and  spawn_positions[1] not in positions[1]:
                        positions[1] = np.sort(np.concatenate((positions[1],[spawn_positions[1]])))
                        dist[1] = positions[1][1:] - positions[1][:-1]
                        N += 1
                    return N, positions, dist, [spawn_position]
        elif case == "multi_HD":
            x = self.kwargs["x_axis"]
            y = self.kwargs["y_axis"]
            spawn_positions = self.kwargs["spawn_pos"]

            n_axis = x.shape[0] + y.shape[0]
            N_spawn = self._growth_ley(N, **self.kwargs)
            rnd_pos = np.random.choice(n_axis, N_spawn)
            for pos in rnd_pos:
                id_1 = int(pos >= x.shape[0])
                id_2 = pos % x.shape[0]
                if spawn_positions[id_1][id_2] != None and spawn_positions[id_1][id_2] not in positions[id_1][id_2]:
                    positions[id_1][id_2] = np.sort(np.concatenate((positions[id_1][id_2],[spawn_positions[id_1][id_2]])))
                    N += 1
                    dist[id_1][id_2] = positions[id_1][id_2][1:] - positions[id_1][id_2][:-1]
            return N, positions, dist, [spawn_positions]



    def _spawn_back(self, positions, dist, case, N):
        if case == "uni":
            N_spawn = self._growth_ley(N, **self.kwargs)
            if N_spawn > 0:
                positions = np.concatenate(([positions[0]-1], positions)).astype(int)
                dist = np.concatenate([[1],dist]).astype(int)
                N += 1
            return N, positions, dist,[positions[0]-1]
        elif case == "multi":
            N_spawn = min(self._growth_ley(N, **self.kwargs), 2)
            match N_spawn:
                case 0:
                    return N, positions, dist,[positions[0]-1]
                case 1:
                    ax = int(np.random.random() < 0.5)
                    positions[ax] = np.concatenate(([positions[ax][0]-1], positions[ax]))
                    dist[ax] = np.concatenate([[1],dist[ax]])
                    N += 1
                    return N, positions, dist,[positions[ax][0]-1]
                case 2:
                    positions[0] = np.concatenate(([positions[0][0]-1], positions[0]))
                    positions[1] = np.concatenate(([positions[1][0]-1], positions[1]))
                    dist[0] = np.concatenate([[1],dist[0]])
                    dist[1] = np.concatenate([[1],dist[1]])
                    N += 2
                    return N, positions, dist,[positions[0][0]-1, positions[1][0]-1]
        elif case == "multi_HD":
            x = self.kwargs["x_axis"]
            y = self.kwargs["y_axis"]
            n_axis = x.shape[0] + y.shape[0]
            N_spawn = self._growth_ley(N, **self.kwargs)
            rnd_pos = np.random.choice(n_axis, max(n_axis,N_spawn), replace=False)
            ret = []
            for pos in rnd_pos:
                id_1 = int(pos >= x.shape[0])
                id_2 = pos % x.shape[0]
                ret.append(positions[id_1][id_2][0] - 1)
                positions[id_1][id_2] = np.concatenate([[positions[id_1][id_2][0] - 1], positions[id_1][id_2]])
                dist[id_1][id_2] = np.concatenate([[1], dist[id_1][id_2]])
                N += 1
            return N, positions, dist, ret
        
    def _spawn_front(self, positions, dist, case, N):
        if case == "uni":
            raise NotImplementedError
        elif case == "multi":
            raise NotImplementedError
        elif case == "multi_HD":
            raise NotImplementedError

        

    def _choose_func(self):
        match self.ley_name:
            case "stable":
                return self._stable_func
            case "spawn_left":
                return self._spwan_left_func
            case "spawn_between":
                return self._spawn_between_func
            case "spawn_fixed":
                return self._spawn_fixed
            case "spawn_front":
                return self._spawn_front   
            case "spawn_back":
                return self._spawn_back

            
    def _stable_func(self, positions, dist, case, N):
        if case == "uni":
            return N, positions, dist, 0
        elif case == "multi":
            return N, positions, dist, 0
        elif case == "multi_HD":
            return N, positions, dist, 0

    def __call__(self, positions, dist, case, N):
        return self.call_func(positions, dist, case, N)