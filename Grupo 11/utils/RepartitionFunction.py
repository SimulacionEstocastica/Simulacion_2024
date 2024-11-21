import numpy as np
import matplotlib.pyplot as plt
import time

class Repartition_Function:
    def __init__(self, case):
        self.case = case

        self.plot_func = self._choose_plot_func()

    def _calc_counts_uni(self, sequence_list):
        x_min = int(np.min(np.concatenate(sequence_list)))
        x_max = int(np.max(np.concatenate(sequence_list)))
        count_arr = np.zeros(x_max-x_min+1)
        for k in range(len(sequence_list)):
            vals, counts = np.unique(sequence_list[k], return_counts = True)
            count_arr[vals.astype(int)-x_min] += counts
        return x_min, x_max, count_arr
    
    def _calc_counts_multi(self, sequence_list):
        x_min_x = int(np.min(np.concatenate([seq[0] for seq in sequence_list])))
        x_max_x = int(np.max(np.concatenate([seq[0] for seq in sequence_list])))
        x_min_y = int(np.min(np.concatenate([seq[1] for seq in sequence_list])))
        x_max_y = int(np.max(np.concatenate([seq[1] for seq in sequence_list])))
        count_arr_x = np.zeros(x_max_x-x_min_x+1)
        count_arr_y = np.zeros(x_max_y-x_min_y+1)
        for k in range(len(sequence_list)):
            vals, counts = np.unique(sequence_list[k][0], return_counts = True)
            count_arr_x[vals.astype(int)-x_min_x] += counts
            vals, counts = np.unique(sequence_list[k][1], return_counts = True)
            count_arr_y[vals.astype(int)-x_min_y] += counts

        return x_min_x, x_min_y, x_max_x, x_max_y, count_arr_x, count_arr_y
    
    def _calc_counts_multi_HD(self, sequence_list):
        n_axis_x = len(sequence_list[0][0])
        n_axis_y = len(sequence_list[0][0])
        bounds_x = np.empty((n_axis_x,2), dtype=int)
        bounds_y = np.empty((n_axis_y,2), dtype=int)
        count_arr_x = []
        count_arr_y = []
        for k in range(n_axis_x):
            x_min_x = int(np.min(np.concatenate([seq[0][k] for seq in sequence_list])))
            x_max_x = int(np.max(np.concatenate([seq[0][k] for seq in sequence_list])))
            bounds_x[k] = [x_min_x, x_max_x]
            count_arr_x.append(np.zeros(x_max_x - x_min_x + 1))

        for k in range(n_axis_y):
            x_min_x = int(np.min(np.concatenate([seq[1][k] for seq in sequence_list])))
            x_max_x = int(np.max(np.concatenate([seq[1][k] for seq in sequence_list])))
            bounds_y[k] = [x_min_x, x_max_x]
            count_arr_y.append(np.zeros(x_max_x - x_min_x + 1))

        for k in range(len(sequence_list)):
            for i in range(n_axis_x):
                vals, counts = np.unique(sequence_list[k][0][i], return_counts = True)
                count_arr_x[i][vals.astype(int)-bounds_x[i,0]] += counts
            for i in range(n_axis_y):
                vals, counts = np.unique(sequence_list[k][1][i], return_counts = True)
                count_arr_y[i][vals.astype(int)-bounds_y[i,0]] += counts

        return bounds_x, bounds_y, count_arr_x, count_arr_y


    def _plot_uni(self, sequence_list, save_path):
        x_min, x_max, count_arr = self._calc_counts_uni(sequence_list)
        plt.scatter(np.arange(x_min, x_max+1), count_arr)
        plt.xlabel("Case number")
        plt.ylabel("Number of particles that went through this case")
        plt.legend()
        plt.grid()
        plt.savefig(save_path, dpi = 300)
        plt.close()


    def _plot_multi(self, sequence_list, save_path):
        x_min_x, x_min_y, x_max_x, x_max_y, count_arr_x, count_arr_y = self._calc_counts_multi(sequence_list)
        f, axs = plt.subplots(1,2,figsize=(10,7))
        #axs[0].subplot(1,2,1)
        axs[0].scatter(np.arange(x_min_x, x_max_x+1), count_arr_x, label = "Axis x")
        axs[0].set_xlabel("Case number")
        axs[0].set_ylabel("Number of particles that went through this case")
        axs[0].legend()
        axs[0].grid()
        #axs[1].subplot(1,2,2)
        axs[1].scatter(np.arange(x_min_y, x_max_y+1), count_arr_y, label = "Axis y")
        axs[1].set_xlabel("Case number")
        axs[1].set_ylabel("Number of particles that went through this case")
        axs[1].legend()
        axs[1].grid()

        plt.savefig(save_path, dpi = 300)
        plt.close()

    def _plot_multi_HD(self, sequence_list, save_path):
        bounds_x, bounds_y, count_arr_x, count_arr_y = self._calc_counts_multi_HD(sequence_list)
        n_axis_x = bounds_x.shape[0]
        n_axis_y = bounds_y.shape[0]

        fig, ax = plt.subplots(figsize = (10,4*max(n_axis_x, n_axis_y)))
        
        for k in range(n_axis_x):
            plt.subplot(max(n_axis_x, n_axis_y), 2, 2*k+1)
            plt.scatter(np.arange(bounds_x[k,0], bounds_x[k,1]+1), count_arr_x[k], label = "Axis x - {}".format(k+1))
            plt.xlabel("Case number")
            plt.ylabel("Number of particles that went through this case")
            plt.legend()
            plt.grid()

        for k in range(n_axis_y):
            plt.subplot(max(n_axis_x, n_axis_y), 2, 2*(k+1))
            plt.scatter(np.arange(bounds_y[k,0], bounds_y[k,1]+1), count_arr_y[k], label = "Axis y - {}".format(k+1))
            plt.xlabel("Case number")
            plt.ylabel("Number of particles that went through this case")
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.savefig(save_path, dpi = 300)
        plt.close()
        

    def _choose_plot_func(self):
        if self.case == "uni":
            return self._plot_uni
        elif self.case == "multi":
            return self._plot_multi
        elif self.case == "multi_HD":
            return self._plot_multi_HD

    def __call__(self, sequence_list, save_path):
        self.plot_func(sequence_list, save_path)

class RepartitionFuncion2:
    def __init__(self, case):
        self.case = case
        self.plot_func = self._choose_plot_func()

    def _find_pos(self, old_frame, new_frame):
        N = len(new_frame) - len(old_frame)
        i1 = 0; i2 = 0
        out = np.empty(N)
        count = 0
        while count < N:
            if new_frame[i1] - old_frame[i2] > 1:
                pass

            
    def _plot_uni(self, sequence_list, save_path):
        x_min = np.min(sequence_list); x_max = np.max(sequence_list)
        altitude = np.zeros((x_max - x_min + 1, len(sequence_list[-1])))
        current_size = len(sequence_list[0])
        index = np.arange(current_size)
        for frame in sequence_list:
            if len(frame) == current_size:
                altitude[frame, index] = True
            else:
                pass
        
    
    def _plot_multi(self, sequence_list, save_path):
        raise NotImplementedError
    
    def _plot_multi_HD(self, sequence_list, save_path):
        raise NotImplementedError

    def _choose_plot_func(self):
        if self.case == "uni":
            return self._plot_uni
        elif self.case == "multi":
            return self._plot_multi
        elif self.case == "multi_HD":
            return self._plot_multi_HD

    def __call__(self, sequence_list, save_path):
        self.plot_func(sequence_list, save_path)