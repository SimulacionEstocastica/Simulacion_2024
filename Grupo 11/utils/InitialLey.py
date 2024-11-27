import numpy as np

class Initial_Ley:
    def __init__(self, ley, x_coordinates, y_coordinates, sparsity = 100) -> None:
        self.ley_name = ley
        self.x_coord = np.array(x_coordinates)
        self.y_coord = np.array(y_coordinates)
        self.sparsity = sparsity
        self._choose_ley()
             

    def _uniform_uni(self, N):
        return np.sort(np.random.choice(np.arange(-self.sparsity*N, self.sparsity*N),N, replace=False))


    def _uniform_multi(self, N):
        # Only for two axes for now
        L = []
        n_x = np.random.binomial(N, 0.5)
        L.append(np.random.choice(np.arange(-self.sparsity*N, self.sparsity*N), n_x, replace=False))
        if (self.y_coord[0] in L[0]):
            L.append(np.random.choice(np.concatenate([np.arange(-self.sparsity*N, 0), np.arange(1,self.sparsity*N)]), N-n_x, replace=False))
        else:
            L.append(np.random.choice(np.arange(-self.sparsity*N, self.sparsity*N), N-n_x, replace=False))
        L[0] = np.sort(L[0])
        L[1] = np.sort(L[1])
        return L

    def _approx_multi(self, N):
        
        n_axis = self.x_coord.shape[0] + self.y_coord.shape[0]
        assert N % n_axis == 0, "Give a multiple of the number of axis please"
        count = 0
        
        L = [[], []]
        for k in range(self.x_coord.shape[0]):
            arr = np.concatenate([np.arange(-self.sparsity*N, self.x_coord[k]), np.arange(self.x_coord[k],self.sparsity*N)])
            L[0].append(np.sort(np.random.choice(arr,N//n_axis, replace=False)))
        for k in range(self.y_coord.shape[0]):
            arr = np.concatenate([np.arange(-self.sparsity*N, self.y_coord[k]), np.arange(self.y_coord[k],self.sparsity*N)])
            L[1].append(np.sort(np.random.choice(arr,N//n_axis, replace=False)))

        return L

    def _choose_ley(self):
        match self.ley_name:
            case "uniform_uni":
                self.call_func = self._uniform_uni
            case "uniform_multi":
                self.call_func = self._uniform_multi
            case "approx_multiHD":
                self.call_func = self._approx_multi

    def __call__(self, N, x_coordinates, y_coordinates):
        
        assert np.all(self.x_coord == x_coordinates) and np.all(self.y_coord == y_coordinates) and len(self.x_coord) == len(x_coordinates) and len(self.y_coord) == len(y_coordinates), "Axis of the problem changed"
        return self.call_func(N)
