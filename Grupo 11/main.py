import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import cv2
import tqdm
from utils.VideoWriter import *
from utils.RepartitionFunction import *
from utils.LeyNumberParticules import *
from utils.MoveLey import *
from utils.InitialLey import *
from utils.Grid import *


if __name__ == '__main__':
    
    ### Case Unidimensional
    """
    N_particles_init = 2**5
    N_steps = 10**2
    x_coords = np.array([0])
    y_coords = np.array([])
    p = np.array(0.2)
    q = np.array(0.1)
    sparsity = 2
    initial_ley = Initial_Ley("uniform_uni", x_coordinates=x_coords, y_coordinates=y_coords, sparsity=sparsity)
    move_ley = Move_Ley("bernoulli_right_sequential_uni", p, q = q) 
    spawn_pos = 5
    params = {"p": 0.01, "spawn_pos":spawn_pos}
    size_N = Ley_Number_Particules("spawn_back", "Bernoulli", kwargs=params)
    obj = Grid(N_particles_init, initial_ley, N_steps, move_ley, size_N, x_lines=x_coords, y_lines=y_coords, simul_type="sequential", verif_mode=True)
    obj.run_simulation()

    rep_func = Repartition_Function("uni")
    rep_func(obj.positions_record, "img/test.png".format(N_steps, sparsity, p, 1-p-0.1))
    #VW = VideoWriter("uni", x_coords, y_axis=y_coords)
    #VW("vid_right_1D.mp4", obj.positions_record)

    ffmpeg_dir = r"C:/Users/antoine\Downloads/ffmpeg-2024-11-03-git-df00705e00-full_build/ffmpeg-2024-11-03-git-df00705e00-full_build/bin/ffmpeg.exe"
    #VW_rep = VideoWriterRepartition("uni", x_coords,ffmpeg_dir, y_axis=y_coords)
    #VW_rep("film_uni.mp4", obj.positions_record)
    """
    ### Case Multidimensional
    """
    N_particles_init = 2**5
    N_steps = 10**2
    x_coords = np.array([0])
    y_coords = np.array([0])
    p = [0.25, 0.55]
    q = [0.15, 0.1]
    sparsity = 2
    cross_axes_prob = [1/3, 1/3, 1/3]
    initial_ley = Initial_Ley("uniform_multi", x_coordinates=x_coords, y_coordinates=y_coords, sparsity=sparsity)
    move_ley = Move_Ley("bernoulli_both_sequential_multi", p, q = q, cross_axes_prob = cross_axes_prob) 
    spawn_pos = [None, None]
    params = {"p": 0.1, "x_axis" : x_coords, "y_axis" : y_coords, "spawn_pos": spawn_pos}
    
    size_N = Ley_Number_Particules("spawn_back", "Bernoulli", kwargs=params)
    #parameters: "spawn_between", "stable"; "Binomial", Bernoulli
    obj = Grid(N_particles_init, initial_ley, N_steps, move_ley, size_N, x_lines=x_coords, y_lines=y_coords, simul_type="sequential", verif_mode=True)
    obj.run_simulation()
    
    rep_func = Repartition_Function("multi")
    rep_func(obj.positions_record, "test_rep.png")
    
    #VW = VideoWriter("multi", x_coords, y_axis=y_coords)
    #VW("test_vid.mp4", obj.positions_record)

    ffmpeg_dir = r"C:/Users/antoine\Downloads/ffmpeg-2024-11-03-git-df00705e00-full_build/ffmpeg-2024-11-03-git-df00705e00-full_build/bin/ffmpeg.exe"
    #VW_rep = VideoWriterRepartition("multi", x_coords, ffmpeg_dir, y_axis=y_coords)
    #VW_rep("film_uni.mp4", obj.positions_record)
    """

    ### Case multidimensional with high number of dimensions
    
    N_particles_init = 2**4
    N_steps = 10**2
    x_coords = np.array([0, 10])
    y_coords = np.array([5,20])
    p = np.array([[0.2,0.5], [0.7, 0.05]])
    q = np.array([[0.2,0.5], [0.1, 0.6]])
    sparsity = 1
    initial_ley = Initial_Ley("approx_multiHD", x_coordinates=x_coords, y_coordinates=y_coords, sparsity=sparsity)
    move_ley = Move_Ley("bernoulli_both_sequential_HD", p, q = q) 
    
    spawn_pos = np.array([[5,None], [None, -3]])
    params = {"p": 0.1, "x_axis" : x_coords, "y_axis" : y_coords, "spawn_pos": spawn_pos}

    size_N = Ley_Number_Particules("spawn_back", "Bernoulli", kwargs=params)
    obj = Grid(N_particles_init, initial_ley, N_steps, move_ley, size_N, x_lines=x_coords, y_lines=y_coords, simul_type="sequential", verif_mode=True)
    obj.run_simulation()

    
    rep_func = Repartition_Function("multi_HD")
    rep_func(obj.positions_record, "img/test.png".format(N_steps, sparsity, p, 1-p-0.1))
    
    #VW = VideoWriter("multi_HD", x_coords, y_axis=y_coords)
    #VW("vid_right_HD.mp4", obj.positions_record)

    ffmpeg_dir = r"C:/Users/antoine\Downloads/ffmpeg-2024-11-03-git-df00705e00-full_build/ffmpeg-2024-11-03-git-df00705e00-full_build/bin/ffmpeg.exe"
    #VW_rep = VideoWriterRepartition("multi_HD", x_coords, ffmpeg_dir, y_axis=y_coords)
    #VW_rep("vid_rep_both_HD.mp4", obj.positions_record)
    
    '''
    x_coords = np.array([0])
    y_coords = np.array([])
    N_particles_init = 2**10
    for N_steps in [10**2,10**5]:
        for sparsity in [1,2,10]:
            for p in [0.2, 0.5, 0.8]:
                initial_ley = Initial_Ley("uniform_uni", x_coordinates=x_coords, y_coordinates=y_coords, sparsity=sparsity)
                move_ley = Move_Ley("bernoulli_right_sequential_uni", p) 
                params = {"p": 0.1, "x_axis": x_coords, "y_axis": y_coords}
                size_N = Ley_Number_Particules("spawn_between", "Bernoulli", kwargs=params)
                obj = Grid(N_particles_init, initial_ley, N_steps, move_ley, size_N, x_lines=x_coords, y_lines=y_coords, simul_type="sequential", verif_mode=False)
                obj.run_simulation()
                
                rep_func = Repartition_Function("uni")
                rep_func(obj.positions_record, "img/right_{}_{}_{}_{}.png".format(N_steps, sparsity, p, 1-p-0.1))


                """
                initial_ley = Initial_Ley("uniform_uni", x_coordinates=x_coords, y_coordinates=y_coords, sparsity=sparsity)
                move_ley = Move_Ley("bernoulli_both_sequential_uni", p, q = 1-p-0.1) 
                params = {"p": 0.1, "x_axis": x_coords, "y_axis": y_coords}
                size_N = Ley_Number_Particules("stable", "Binomial", kwargs=params)
                obj = Grid(N_particles_init, initial_ley, N_steps, move_ley, size_N, x_lines=x_coords, y_lines=y_coords, simul_type="sequential", verif_mode=False)
                obj.run_simulation()

                rep_func = Repartition_Function("uni")
                rep_func(obj.positions_record, "img/both_{}_{}_{}_{}.png".format(N_steps, sparsity, p, 1-p-0.1))
                """
    '''
    """
    x_coords = np.array([0])
    y_coords = np.array([0])
    N_particles_init = 2**10
    for N_steps in [10**2,10**3,10**4,10**5,10**6]:
        for sparsity in [1,2,5,10]:
            for p in [0.1,0.2,0.5,0.7]:
                initial_ley = Initial_Ley("uniform_uni", x_coordinates=x_coords, y_coordinates=y_coords, sparsity=sparsity)
                move_ley = Move_Ley("bernoulli_right_sequential_uni", p) 
                params = {"p": 0.1, "x_axis": x_coords, "y_axis": y_coords}
                size_N = Ley_Number_Particules("stable", "Binomial", kwargs=params)
                obj = Grid(N_particles_init, initial_ley, N_steps, move_ley, size_N, x_lines=x_coords, y_lines=y_coords, simul_type="sequential", verif_mode=False)
                obj.run_simulation()
                
                rep_func = Repartition_Function("uni")
                rep_func(obj.positions_record, "img/right_{}_{}_{}_{}.png".format(N_steps, sparsity, p, 1-p-0.1))

                initial_ley = Initial_Ley("uniform_uni", x_coordinates=x_coords, y_coordinates=y_coords, sparsity=sparsity)
                move_ley = Move_Ley("bernoulli_both_sequential_uni", p, q = 1-p-0.1) 
                params = {"p": 0.1, "x_axis": x_coords, "y_axis": y_coords}
                size_N = Ley_Number_Particules("stable", "Binomial", kwargs=params)
                obj = Grid(N_particles_init, initial_ley, N_steps, move_ley, size_N, x_lines=x_coords, y_lines=y_coords, simul_type="sequential", verif_mode=False)
                obj.run_simulation()

                rep_func = Repartition_Function("uni")
                rep_func(obj.positions_record, "img/both_{}_{}_{}_{}.png".format(N_steps, sparsity, p, 1-p-0.1))
    """