import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tqdm
from matplotlib.animation import FuncAnimation, FFMpegWriter

class VideoWriter:


    def __init__(self, case, x_axis, y_axis = []):
        self.case = case
        self.x = np.array(x_axis)
        self.y = np.array(y_axis)

        self.VW = self._choose_VW()

    def _build_frame_uni(self, frame, block_size_x, block_size_y, borders):
        x_min, x_max = borders
        Delta = x_max - x_min

        grid = np.zeros((int(block_size_x*Delta), int(block_size_y*10)), dtype=np.uint8)
        for pos_x in frame:
            x_beg = int((pos_x - x_min)*block_size_x)
            y_beg = int(4*block_size_y)
            grid[x_beg:x_beg + block_size_x, y_beg:y_beg+block_size_y] = 255
        return grid

    def _build_frame_multi(self, frame, block_size_x, block_size_y, borders):
        x_min_x, x_max_x, x_min_y, x_max_y = borders
        Delta_x = x_max_x - x_min_x; Delta_y = x_max_y - x_min_y

        grid = np.zeros((int(block_size_x*Delta_x), int(block_size_y*Delta_y)), dtype=np.uint8)
        for pos_x in frame[0]:
            x_beg = int((pos_x - x_min_x)*block_size_x)
            y_beg = int((self.y - x_min_y)*block_size_y)
            grid[x_beg:x_beg + block_size_x, y_beg:y_beg+block_size_y] = 255
        for pos_y in frame[1]:
            y_beg = int((pos_y - x_min_y)*block_size_y)
            x_beg = int((self.x - x_min_x)*block_size_x)
            grid[x_beg:x_beg + block_size_x, y_beg:y_beg+block_size_y] = 255
        return grid

    def _build_frame_multi_HD(self, frame, block_size_x, block_size_y, borders):
        x_min_x, x_max_x, x_min_y, x_max_y = borders
        Delta_x = x_max_x - x_min_x; Delta_y = x_max_y - x_min_y

        grid = np.zeros((int(block_size_x*Delta_x), int(block_size_y*Delta_y)), dtype=np.uint8)
        for k, f in enumerate(frame[0]):
            y_pos = self.y[k]
            for pos_x in f:
                x_beg = int((pos_x - x_min_x)*block_size_x)
                y_beg = int((y_pos - x_min_y)*block_size_y)
                grid[x_beg:x_beg + block_size_x, y_beg:y_beg+block_size_y] = 255
        for k, f in enumerate(frame[1]):
            x_pos = self.x[k]
            for pos_y in f:
                y_beg = int((pos_y - x_min_y)*block_size_y)
                x_beg = int((x_pos - x_min_x)*block_size_x)
                grid[x_beg:x_beg + block_size_x, y_beg:y_beg+block_size_y] = 255
        return grid

    def _WV_uni(self, save_path, sequences, block_size_x = 10, block_size_y = 10, fps = 10):
        x_min = sequences[0][0] 
        x_max = sequences[-1][-1]
        assert x_max - x_min < 500, "Too much particles on x axis!"

        borders = (x_min, x_max)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, fps, (int((x_max-x_min)*block_size_x), 10*block_size_y))
        
        for frame in sequences:
            frame_modified = self._build_frame_uni(frame, block_size_x, block_size_y, borders)
            frame_modified = cv2.applyColorMap(np.transpose(frame_modified), cv2.COLORMAP_JET)
            video.write(frame_modified)

        video.release()
        cv2.destroyAllWindows()

    def _WV_multi(self, save_path, sequences, block_size_x = 10, block_size_y = 10, fps = 10):
        x_min_x = sequences[0][0][0] 
        x_max_x = sequences[-1][0][-1]
        x_min_y = sequences[0][1][0]
        x_max_y = sequences[-1][1][-1]
        assert x_max_x - x_min_x < 500, "Too much particles on x axis!"
        assert x_max_y - x_min_y < 500, "Too much particles on y axis!"

        borders = (x_min_x, x_max_x, x_min_y, x_max_y)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, fps, (int((x_max_x-x_min_x)*block_size_x), int((x_max_y-x_min_y)*block_size_y)))
        
        for frame in sequences:
            frame_modified = self._build_frame_multi(frame, block_size_x, block_size_y, borders)
            frame_modified = cv2.applyColorMap(np.transpose(frame_modified), cv2.COLORMAP_JET)
            video.write(frame_modified)
            #print(frame_modified.shape, (int((x_max_x-x_min_x)*block_size_x), int((x_max_y-x_min_y)*block_size_y)))

        video.release()
        cv2.destroyAllWindows()

    def _WV_multi_HD(self, save_path, sequences, block_size_x = 10, block_size_y = 10, fps = 10):
        x_min_x = np.min(np.concatenate(sequences[0][0]))
        x_max_x = np.max(np.concatenate(sequences[-1][0]))
        x_min_y = np.min(np.concatenate(sequences[0][1]))
        x_max_y = np.max(np.concatenate(sequences[-1][1]))
        assert x_max_x - x_min_x < 500, "Too much particles on x axis!"
        assert x_max_y - x_min_y < 500, "Too much particles on y axis!"

        borders = (x_min_x, x_max_x, x_min_y, x_max_y)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, fps, (int((x_max_x-x_min_x)*block_size_x), int((x_max_y-x_min_y)*block_size_y)))
        
        for frame in sequences:
            frame_modified = self._build_frame_multi_HD(frame, block_size_x, block_size_y, borders)
            frame_modified = cv2.applyColorMap(np.transpose(frame_modified), cv2.COLORMAP_JET)
            video.write(frame_modified)
            #print(frame_modified.shape, (int((x_max_x-x_min_x)*block_size_x), int((x_max_y-x_min_y)*block_size_y)))

        video.release()
        cv2.destroyAllWindows()

    def _choose_VW(self):
        if self.case == "uni":
            return self._WV_uni
        elif self.case == "multi":
            return self._WV_multi
        elif self.case == "multi_HD":
            return self._WV_multi_HD
        
    def __call__(self, save_path, sequences, **kwargs):
        self.VW(save_path, sequences, **kwargs)


class VideoWriterRepartition:
    def __init__(self, case, x_axis, ffmpeg_dir, y_axis = []):
        self.case = case
        self.x = np.array(x_axis)
        self.y = np.array(y_axis)
        self.VW = self._choose_VW()
        self.ffmpeg_dir = ffmpeg_dir

    def _calc_bounds_multi_HD(self, sequence_list):
        n_axis_x = len(sequence_list[0][0])
        n_axis_y = len(sequence_list[0][0])
        bounds_x = np.empty((n_axis_x,2), dtype=int)
        bounds_y = np.empty((n_axis_y,2), dtype=int)
        
        for k in range(n_axis_x):
            x_min_x = np.min(np.concatenate([seq[0][k] for seq in sequence_list]))
            x_max_x = np.max(np.concatenate([seq[0][k] for seq in sequence_list]))
            bounds_x[k] = [x_min_x, x_max_x]

        for k in range(n_axis_y):
            x_min_x = np.min(np.concatenate([seq[1][k] for seq in sequence_list]))
            x_max_x = np.max(np.concatenate([seq[1][k] for seq in sequence_list]))
            bounds_y[k] = [x_min_x, x_max_x]

        return bounds_x, bounds_y

    def _plot_frame_multi_HD(self, frame, repartition_x, repartition_y, cumulated_sum_x, cumulated_sum_y, bounds_x, bounds_y):
        n_axis_x = len(frame[0])
        n_axis_y = len(frame[1])
        for k in range(n_axis_x):
            vals, counts = np.unique(frame[0][k], return_counts = True)
            repartition_x[k][vals-bounds_x[k,0]] += counts
            cumulated_sum_x[k] += np.sum(counts)
            plt.subplot(max(n_axis_x, n_axis_y), 2, 2*k+1)
            plt.scatter(np.arange(bounds_x[k,0], bounds_x[k,1]+1), repartition_x[k]/cumulated_sum_x[k])
            plt.xlabel("Case number")
            plt.ylabel("Number of particles that went through this case")
            plt.legend("Axis x")
            plt.grid()
        
        for k in range(n_axis_y):
            vals, counts = np.unique(frame[1][k], return_counts = True)
            repartition_y[k][vals-bounds_y[k,0]] += counts
            cumulated_sum_y[k] += np.sum(counts)
            plt.subplot(max(n_axis_x, n_axis_y), 2, 2*(k+1))
            plt.scatter(np.arange(bounds_y[k,0], bounds_y[k,1]+1), repartition_y[k]/cumulated_sum_y[k])
            plt.xlabel("Case number")
            plt.ylabel("Number of particles that went through this case")
            plt.legend("Axis x")
            plt.grid()
        
        plt.tight_layout()
        plt.savefig("img/transitory_img.png", dpi = 300)
        plt.clear()


        return repartition_x, repartition_y, cumulated_sum_x, cumulated_sum_y

    def _VW_uni(self, save_path, sequences, fps = 50):
        bounds = np.array([np.min(sequences), np.max(sequences)]).astype(int)
        repartition = np.zeros(bounds[1] - bounds[0] + 1)
        global cumulated_sum
        cumulated_sum = 0

        fig, ax = plt.subplots()
        plot_vals = ax.plot(np.arange(bounds[0], bounds[1] + 1), repartition)[0]

        def update(frame):
            global cumulated_sum
            vals, counts = np.unique(frame, return_counts = True)
            repartition[vals.astype(int) - bounds[0]] += counts
            cumulated_sum += np.sum(counts)
            plot_vals.set_ydata(repartition/cumulated_sum)
            ax.set_ylim(0, np.max(repartition/cumulated_sum))
            return plot_vals,
        
        plt.rcParams['animation.ffmpeg_path'] = self.ffmpeg_dir #r"C:/Users/antoine\Downloads/ffmpeg-2024-11-03-git-df00705e00-full_build/ffmpeg-2024-11-03-git-df00705e00-full_build/bin/ffmpeg.exe"
        FFwriter = FFMpegWriter(fps=fps, codec="h264")
        ani = FuncAnimation(fig, update, frames=sequences, interval=20, blit=True)
        ani.save(save_path,writer = FFwriter, dpi = 300)


    def _VW_multi(self, save_path, sequences, fps = 50):
        P1 = np.concatenate([seq[0] for seq in sequences])
        P2 = np.concatenate([seq[1] for seq in sequences])
        bounds_x = np.array([np.min(P1), np.max(P1)]).astype(int)
        bounds_y =np.array([np.min(P2), np.max(P2)]).astype(int)

        repartition_x = np.zeros(bounds_x[1] - bounds_x[0] + 1)
        repartition_y = np.zeros(bounds_y[1] - bounds_y[0] + 1)

        cumulated_sum = np.array([0,0])

        fig, ax = plt.subplots(1,2)
        S = [ax[0].plot(np.arange(bounds_x[0], bounds_x[1] + 1), repartition_x)[0], ax[1].plot(np.arange(bounds_y[0], bounds_y[1] + 1), repartition_y)[0]]

        def update(frame):
            vals, counts = np.unique(frame[0], return_counts = True)
            repartition_x[vals.astype(int) - bounds_x[0]] += counts
            cumulated_sum[0] += np.sum(counts)
            S[0].set_ydata(repartition_x/cumulated_sum[0])
            ax[0].set_ylim(0, np.max(repartition_x/cumulated_sum[0]))

            vals, counts = np.unique(frame[1], return_counts = True)
            repartition_y[vals.astype(int) - bounds_y[0]] += counts
            cumulated_sum[1] += np.sum(counts)
            S[1].set_ydata(repartition_y/cumulated_sum[1])
            ax[1].set_ylim(0, np.max(repartition_y/cumulated_sum[1]))
            return S
        

        plt.rcParams['animation.ffmpeg_path'] = self.ffmpeg_dir #r"C:/Users/antoine\Downloads/ffmpeg-2024-11-03-git-df00705e00-full_build/ffmpeg-2024-11-03-git-df00705e00-full_build/bin/ffmpeg.exe"
        FFwriter = FFMpegWriter(fps=fps, codec="h264")
        ani = FuncAnimation(fig, update, frames=sequences, interval=20, blit=True)
        ani.save(save_path,writer = FFwriter, dpi = 300)

    def _VW_multi_HD(self, save_path, sequences, fps = 50):
        bounds_x, bounds_y = self._calc_bounds_multi_HD(sequences)
        repartition_x = [np.zeros(bounds_x[k,1] - bounds_x[k,0] + 1) for k in range(len(bounds_x))]
        repartition_y = [np.zeros(bounds_y[k,1] - bounds_y[k,0] + 1) for k in range(len(bounds_y))]

        cumulated_sum_x = np.zeros(len(bounds_x))
        cumulated_sum_y = np.zeros(len(bounds_y))

        n_axis_x = bounds_x.shape[0]
        n_axis_y = bounds_y.shape[0]

        fig, axs = plt.subplots(max(n_axis_x, n_axis_y), 2)

        S2_x = []
        S2_y = []

        for k in range(n_axis_x):
            axs[k,0].set_xlim(bounds_x[k,0],bounds_x[k,1])
            axs[k,0].set_ylim(0,0.2)
            S2_x.append(axs[k, 0].plot(np.arange(bounds_x[k,0], bounds_x[k,1]+1), np.zeros(bounds_x[k,1] - bounds_x[k,0]+1))[0])

        for k in range(n_axis_y):
            axs[k,1].set_xlim(bounds_y[k,0],bounds_y[k,1])
            axs[k,1].set_ylim(0,0.2)
            S2_y.append(axs[k, 1].plot(np.arange( bounds_y[k,0],bounds_y[k,1]+1), np.zeros(bounds_y[k,1] - bounds_y[k,0]+1))[0])
        
        def update(frame):
            for k in range(n_axis_x):
                vals, counts = np.unique(frame[0][k], return_counts = True)
                repartition_x[k][vals.astype(int)-bounds_x[k,0]] += counts
                cumulated_sum_x[k] += np.sum(counts)
                
                axs[k,0].set_ylim(0, np.max(repartition_x[k] / cumulated_sum_x[k]))
                
                S2_x[k].set_ydata(repartition_x[k] / cumulated_sum_x[k])
                
            for k in range(n_axis_y):
                vals, counts = np.unique(frame[1][k], return_counts = True)
                repartition_y[k][vals.astype(int)-bounds_y[k,0]] += counts
                cumulated_sum_y[k] += np.sum(counts)
                
                axs[k,1].set_ylim(0, np.max(repartition_y[k] / cumulated_sum_y[k]))
                S2_y[k].set_ydata(repartition_y[k] / cumulated_sum_y[k])
            return np.concatenate([S2_x, S2_y])
        
        plt.rcParams['animation.ffmpeg_path'] = self.ffmpeg_dir #r"C:/Users/antoine\Downloads/ffmpeg-2024-11-03-git-df00705e00-full_build/ffmpeg-2024-11-03-git-df00705e00-full_build/bin/ffmpeg.exe"
        FFwriter = FFMpegWriter(fps=fps, codec="h264")
        ani = FuncAnimation(fig, update, frames=sequences, interval=20, blit=True)
        ani.save(save_path,writer = FFwriter, dpi = 300)

    def _choose_VW(self):
        if self.case == "uni":
            return self._VW_uni
        elif self.case == "multi":
            return self._VW_multi
        elif self.case == "multi_HD":
            return self._VW_multi_HD
                    

    def __call__(self, save_path, sequences, fps = 10):
        self.VW(save_path, sequences, fps)