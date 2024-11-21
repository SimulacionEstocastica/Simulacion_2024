import numpy as np

class Move_Ley:
    def __init__(self, ley, p, q = [], cross_axes_prob = []):
        self.ley_name = ley
        self.p = np.array(p)
        self.q = np.array(q)
        self.CAP = np.array(cross_axes_prob)
        self.call_func = self._choose_func()
        self._move_cat = self._move_type()

    def _verify_problem(self):
        assert not(self.direction == "right" and self.q.shape[0] > 0), "Provided a q when only going to the right"
        assert not(self.direction == "both" and self.q.shape[0] == 0), "Didn't provide a q when going both sides"
        assert not(self._move_cat[1] == "uni" and (self.q.shape[0] > 1 or self.p.shape[0] > 1)), "Provided vector of probabilities p or q when unidimensional problem"
        assert not(self._move_cat[1] == "multi" and self.p.shape[0] == 1), "Didn't provide a vector of probabilities p when multidimensional problem"
        assert not(self._move_cat[1] == "multi" and self.q.shape[0] == 1), "Didn't provide a vector of probabilities q when multidimensional problem"
        assert np.sum(self.CAP) == 1


    def _bernoulli_right_sequential_uni(self, positions, dist, **kwargs):

        mask = np.zeros(shape = positions.shape[0], dtype=np.bool_)
        mask_2 = np.zeros(shape = positions.shape[0], dtype=np.bool_)

        neighbours_forward = (dist == 1).astype(np.bool_)
        u = (np.random.random(size = positions.shape[0]) < self.p).astype(np.bool_)
        mask[:-1] = np.logical_and(u[:-1], np.logical_not(neighbours_forward))
        mask[-1] = u[-1]
        positions[mask] += 1
        mask_2[:-1] = np.logical_and(np.logical_and(u[:-1], neighbours_forward), mask[:-1])
        positions[mask_2] += 1


        mask_f = np.logical_or(mask, mask_2)
        prod =  np.logical_and(mask_f, u).astype(int)
        dist += prod[1:] - prod[:-1]

        return positions, dist
        



    def _bernoulli_both_sequential_uni(self, positions, dist, **kwargs):

        
        u = np.random.choice([-1,0,1], size = positions.shape[0], p=[self.q, 1 - self.p - self.q,self.p])
        mask_u = (u == 1).astype(np.bool_)
        positions -= (u==-1)
        while True:
            mask = (positions[1:] == positions[:-1]).astype(np.bool_)
            if not mask.any():
                break
            else:
                positions[:-1][mask] -= 1

        dist = positions[1:] - positions[:-1]
        neighbours = (dist == 1).astype(np.bool_)
        mask = np.zeros(positions.shape[0], dtype=np.bool_)
        mask[-1] = mask_u[-1]
        mask[:-1] = np.logical_and(np.logical_not(neighbours), mask_u[:-1])
        mask[:-1]= np.logical_or(mask[:-1], np.logical_and(mask[1:], mask_u[:-1]))

        positions += mask 
        dist = positions[1:] - positions[:-1]

        return positions, dist

    def _bernoulli_right_parallel_uni(self, positions, dist, **kwargs):
        raise NotImplementedError

    def _bernoulli_both_parallel_uni(self, positions, dist, **kwargs):
        raise NotImplementedError

    def _bernoulli_right_sequential_multi(self, positions, dist, **kwargs):
        ## For now: only 1 x and y axis
        
        x_axis = kwargs["x_axis"]
        y_axis = kwargs["y_axis"]
        x = x_axis[0]; y = y_axis[0]
        n_x = positions[0].shape[0]
        n_y = positions[1].shape[0]
        N = n_x+n_y
        u_x = np.random.random(size=n_x)
        prob_x = (u_x < self.p[0]).astype(np.bool_)
        u_y = np.random.random(size=n_y)
        prob_y = (u_y < self.p[1]).astype(np.bool_)

        neighbours_forward_x = (dist[0] == 1).astype(np.bool_)
        neighbours_forward_y = (dist[1] == 1).astype(np.bool_)

        #Right part
        mask = np.zeros(shape = n_x, dtype=np.bool_)
        
        mask[-1] = prob_x[-1]
        mask[:-1] = np.logical_and(prob_x[:-1], np.logical_not(neighbours_forward_x))
        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_x, np.logical_and(mask[1:], prob_x[:-1])))
        mask[positions[0] <= x] = False #Choose the particules on the right side

        positions[0] += mask #These particles go right
        

        #Down part
        mask = np.zeros(shape = n_y, dtype=np.bool_)

        mask[-1] = prob_y[-1]
        mask[:-1] = np.logical_and(prob_y[:-1], np.logical_not(neighbours_forward_y))
        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_y, np.logical_and(mask[1:], prob_y[:-1])))
        mask[positions[1] <= y] = False #Choose the particules on the right side

        positions[1] += mask #These particles go right

        dist[0] = positions[0][1:] - positions[0][:-1]
        dist[1] = positions[1][1:] - positions[1][:-1]
        neighbours_forward_x = (dist[0] == 1).astype(np.bool_)
        neighbours_forward_y = (dist[1] == 1).astype(np.bool_)

        #Mid point
        flag_x = True
        flag_y = True
        ind_x = np.where(positions[0] == y)[0] #Check if a particle is at the cross and belongs to x axis
        ind_y = np.where(positions[1] == x)[0] #Check if a particle is at the cross and belongs to y axis
        
        if ind_x.shape[0] == 1:
            flag_x = False
            #If the particle is on the x axis
            
            if prob_x[ind_x] and (ind_x == n_x  or neighbours_forward_x[ind_x] > 1):
                #if A and B; A: if the particle has to move (binomial) on the x axis; B: the particle souldn't have a forward neighbour on the x axis or is the first one on this axis
                flag_x = True
                positions[0][ind_x] += 1
                
            elif u_x[ind_x] < self.CAP[0] + self.CAP[1] and x+1 not in positions[1]:
                
                flag_x = True
                mask_exclusion = np.ones(n_x).astype(np.bool_)
                mask_exclusion[ind_x] = False #The new x axis is like the previous one without the particle
                positions[0] = positions[0][mask_exclusion] #Particle excluded
                idy = np.searchsorted(positions[1], x+1) #Find the index in the sorted array of y particles to insert the particle
                positions[1] = np.insert(positions[1], idy, x+1) #Particle inserted in the y axis
                u_y = np.insert(u_y, idy, u_x[ind_x]) #The random probability that corresponds to the particle is inserted in the vector of random prob. of the y axos
                prob_y = np.insert(prob_y, idy, prob_x[ind_x])
                u_x = u_x[mask_exclusion] #This probability is deleted from the x axis vector of prob.
                prob_x = prob_x[mask_exclusion]
                n_x -= 1; n_y +=1 #Number of particles in both axis is updated
                dist[0] = positions[0][1:] - positions[0][:-1] #Distances between particles is updated on x axis
                dist[1] = positions[1][1:] - positions[1][:-1] #Distances between particles is updated on y axis
                neighbours_forward_x = dist[0] == 1 #New positions for neighbours is updated
                neighbours_forward_y = dist[1] == 1 #New positions for neighbours is updated
        elif ind_y.shape[0] == 1:
            flag_y = False
            if prob_y[ind_y] and (ind_y == n_y  or neighbours_forward_y[ind_y] > 1):
                positions[1][ind_y] += 1
                flag_y = True
                
            elif u_y[ind_y] < self.CAP[0] + self.CAP[1] and y+1 not in positions[0]:
                
                flag_y = True
                mask_exclusion = np.ones(n_y).astype(np.bool_)
                mask_exclusion[ind_y] = False
                positions[1] = positions[1][mask_exclusion]
                idx = np.searchsorted(positions[0], y+1)
                
                positions[0] = np.insert(positions[0], idx, y+1)
                
                u_x = np.insert(u_x, idx, u_y[ind_y])
                prob_x = np.insert(prob_x, idx, prob_y[ind_y])
                u_y = u_y[mask_exclusion]
                prob_y = prob_y[mask_exclusion]
                n_x += 1; n_y -=1
                
                dist[0] = positions[0][1:] - positions[0][:-1]
                dist[1] = positions[1][1:] - positions[1][:-1]
                neighbours_forward_x = dist[0] == 1
                neighbours_forward_y = dist[1] == 1
            
        #Left part
        mask = np.zeros(shape = n_x, dtype=np.bool_)
        
        mask[:-1] = np.logical_and(prob_x[:-1], np.logical_not(neighbours_forward_x))

        global_flag = flag_x and flag_y
        second_flag = False
        ind_prev = np.searchsorted(positions[0], x-1)
        if positions[0][ind_prev] == x-1:
            mask[ind_prev] = global_flag and prob_x[ind_prev]
            second_flag = prob_x[ind_prev]

        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_x, np.logical_and(mask[1:], prob_x[:-1])))
        mask[positions[0] >= x] = False #Choose the particules on the right side
        

        positions[0] += mask #These particles go right
        dist[0] = positions[0][1:] - positions[0][:-1]

        #Upper part
        mask = np.zeros(shape = n_y, dtype=np.bool_)
        
        mask[:-1] = np.logical_and(prob_y[:-1], np.logical_not(neighbours_forward_y))
        
        #print()
        ind_prev = np.searchsorted(positions[1], y-1)
        if positions[1][ind_prev] == y-1:
            global_flag = global_flag and not second_flag and prob_y[ind_prev]
            #print(global_flag)
            mask[ind_prev] = global_flag

        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_y, np.logical_and(mask[1:], prob_y[:-1])))
        if positions[1][ind_prev] == y-1:
            global_flag = global_flag and not second_flag and prob_y[ind_prev]
            #print(global_flag)
            mask[ind_prev] = global_flag
        mask[positions[1] >= y] = False #Choose the particules on the right side
        
        

        positions[1] += mask #These particles go right
        dist[1] = positions[1][1:] - positions[1][:-1]

        return positions, dist

    def _bernoulli_both_sequential_multi(self, positions, dist, **kwargs):
        ## For now: only 1 x and y axis
        
        x_axis = kwargs["x_axis"]
        y_axis = kwargs["y_axis"]
        x = x_axis[0]; y = y_axis[0]
        n_x = positions[0].shape[0]
        n_y = positions[1].shape[0]
        N = n_x+n_y
        u_x = np.random.choice([-1,0,1], size=n_x, p = [self.q[0], 1 - self.p[0] - self.q[0],self.p[0]])
        prob_x_left = (u_x == -1).astype(np.bool_)
        prob_x_right = (u_x == 1).astype(np.bool_)
        u_y = np.random.choice([-1,0,1], size=n_y, p = [self.q[1], 1 - self.p[1] - self.q[1],self.p[1]])
        prob_y_left = (u_y == -1).astype(np.bool_)
        prob_y_right = (u_y == 1).astype(np.bool_)

        ### X axis
        ### ### Left Move

        ind_y = np.searchsorted(positions[1], x)
        if positions[1][ind_y] == x:
            ind_x = np.searchsorted(positions[0], y)
            mask = np.ones(n_y, dtype=np.bool_)
            mask[ind_y] = False
            positions[0] = np.insert(positions[0], ind_x, x)
            positions[1] = positions[1][mask]
            
            prob_x_left = np.insert(prob_x_left, ind_x, prob_y_left[ind_y])
            prob_x_right = np.insert(prob_x_right, ind_x, prob_y_right[ind_y])
            u_x = np.insert(u_x, ind_x, u_y[ind_y])

            prob_y_left = prob_y_left[mask]
            prob_y_right = prob_y_right[mask]
            u_y = u_y[mask]
            n_x += 1; n_y -= 1

        positions[0] -= prob_x_left
        while True:
            mask = (positions[0][1:] == positions[0][:-1]).astype(np.bool_)
            if np.sum(mask) == 0:
                break
            else:
                positions[0][:-1][mask] -= 1

        ### Y axis
        ### ### Left Move

        ind_x = np.searchsorted(positions[0], y)
        if positions[0][ind_x] == y:
            ind_y = np.searchsorted(positions[1], x)
            mask = np.ones(n_x, dtype=np.bool_)
            mask[ind_x] = False
            positions[1] = np.insert(positions[1], ind_y, y)
            positions[0] = positions[0][mask]
            
            prob_y_left = np.insert(prob_y_left, ind_y, prob_x_left[ind_x])
            prob_y_right = np.insert(prob_y_right, ind_y, prob_x_right[ind_x])
            u_y = np.insert(u_y, ind_y, u_x[ind_x])

            prob_x_left = prob_x_left[mask]
            prob_x_right = prob_x_right[mask]
            u_x = u_x[mask]
            n_y += 1; n_x -= 1

        positions[1] -= prob_y_left
        """
        ind_x = np.searchsorted(positions[0], y)
        ind_y = np.searchsorted(positions[1], x)
        if positions[1][ind_y] == x and positions[0][ind_x] == y:
            positions[1][ind_y] -= 1

        """
        while True:
            mask = (positions[1][1:] == positions[1][:-1]).astype(np.bool_)
            if np.sum(mask) == 0:
                break
            else:
                positions[1][:-1][mask] -= 1

        dist[0] = positions[0][1:] - positions[0][:-1]
        dist[1] = positions[1][1:] - positions[1][:-1]
        neighbours_forward_x = (dist[0] == 1).astype(np.bool_)
        neighbours_forward_y = (dist[1] == 1).astype(np.bool_)

        ### Right part
        ### ### Right Move

        mask = np.zeros(shape = n_x, dtype=np.bool_)
        
        mask[-1] = prob_x_right[-1]
        mask[:-1] = np.logical_and(prob_x_right[:-1], np.logical_not(neighbours_forward_x))
        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_x, np.logical_and(mask[1:], prob_x_right[:-1])))
        mask[positions[0] <= x] = False #Choose the particules on the right side

        positions[0] += mask #These particles go right

        ### Bottom Part
        ### ### Right Move

        mask = np.zeros(shape = n_y, dtype=np.bool_)

        mask[-1] = prob_y_right[-1]
        mask[:-1] = np.logical_and(prob_y_right[:-1], np.logical_not(neighbours_forward_y))
        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_y, np.logical_and(mask[1:], prob_y_right[:-1])))
        mask[positions[1] <= y] = False #Choose the particules on the right side

        positions[1] += mask #These particles go right

        dist[0] = positions[0][1:] - positions[0][:-1]
        dist[1] = positions[1][1:] - positions[1][:-1]
        neighbours_forward_x = (dist[0] == 1).astype(np.bool_)
        neighbours_forward_y = (dist[1] == 1).astype(np.bool_)

        ### Central Part

        flag_x = True
        flag_y = True
        ind_x = np.where(positions[0] == y)[0] #Check if a particle is at the cross and belongs to x axis
        ind_y = np.where(positions[1] == x)[0] #Check if a particle is at the cross and belongs to y axis
        if ind_x.shape[0] == 1:
            flag_x = False
            #If the particle is on the x axis
            if prob_x_right[ind_x] and (ind_x == n_x  or neighbours_forward_x[ind_x] > 1):
                #if A and B; A: if the particle has to move (binomial) on the x axis; B: the particle souldn't have a forward neighbour on the x axis or is the first one on this axis
                flag_x = True
                positions[0][ind_x] += 1
            elif u_x[ind_x] < self.CAP[0] + self.CAP[1] and x+1 not in positions[1]:
                flag_x = True
                mask_exclusion = np.ones(n_x).astype(np.bool_)
                mask_exclusion[ind_x] = False #The new x axis is like the previous one without the particle
                positions[0] = positions[0][mask_exclusion] #Particle excluded
                idy = np.searchsorted(positions[1], x+1) #Find the index in the sorted array of y particles to insert the particle
                positions[1] = np.insert(positions[1], idy, x+1) #Particle inserted in the y axis
                u_y = np.insert(u_y, idy, u_x[ind_x]) #The random probability that corresponds to the particle is inserted in the vector of random prob. of the y axos
                prob_y_right = np.insert(prob_y_right, idy, prob_x_right[ind_x])
                u_x = u_x[mask_exclusion] #This probability is deleted from the x axis vector of prob.
                prob_x_right = prob_x_right[mask_exclusion]
                n_x -= 1; n_y +=1 #Number of particles in both axis is updated
                dist[0] = positions[0][1:] - positions[0][:-1] #Distances between particles is updated on x axis
                dist[1] = positions[1][1:] - positions[1][:-1] #Distances between particles is updated on y axis
                neighbours_forward_x = dist[0] == 1 #New positions for neighbours is updated
                neighbours_forward_y = dist[1] == 1 #New positions for neighbours is updated
        elif ind_y.shape[0] == 1:
            flag_y = False
            if prob_y_right[ind_y] and (ind_y == n_y  or neighbours_forward_y[ind_y] > 1):
                positions[1][ind_y] += 1
                flag_y = True
            elif u_y[ind_y] < self.CAP[0] + self.CAP[1] and y+1 not in positions[0]:
                flag_y = True
                mask_exclusion = np.ones(n_y).astype(np.bool_)
                mask_exclusion[ind_y] = False
                positions[1] = positions[1][mask_exclusion]
                idx = np.searchsorted(positions[0], y+1)

                positions[0] = np.insert(positions[0], idx, y+1)
                
                u_x = np.insert(u_x, idx, u_y[ind_y])
                prob_x_right = np.insert(prob_x_right, idx, prob_y_right[ind_y])
                u_y = u_y[mask_exclusion]
                prob_y_right = prob_y_right[mask_exclusion]
                n_x += 1; n_y -=1
                
                dist[0] = positions[0][1:] - positions[0][:-1]
                dist[1] = positions[1][1:] - positions[1][:-1]
                neighbours_forward_x = dist[0] == 1
                neighbours_forward_y = dist[1] == 1

        ### Left Part
        ### ### Right Move

        mask = np.zeros(shape = n_x, dtype=np.bool_)
        
        mask[:-1] = np.logical_and(prob_x_right[:-1], np.logical_not(neighbours_forward_x))

        global_flag = flag_x and flag_y
        second_flag = False
        ind_prev = np.searchsorted(positions[0], x-1)
        if positions[0][ind_prev] == x-1:
            mask[ind_prev] = global_flag and prob_x_right[ind_prev]
            second_flag = prob_x_right[ind_prev]


        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_x, np.logical_and(mask[1:], prob_x_right[:-1])))
        mask[positions[0] >= x] = False #Choose the particules on the right side
        """
        global_flag = flag_x and flag_y
        second_flag = False
        ind_prev = np.searchsorted(positions[0], y-1)
        if positions[0][ind_prev] == y-1:
            mask[ind_prev] = global_flag and mask[ind_prev]
            second_flag = mask[ind_prev]
        """
        positions[0] += mask #These particles go right
        

        ### Upper Part
        ### ### Right Move

        mask = np.zeros(shape = n_y, dtype=np.bool_)
        
        mask[:-1] = np.logical_and(prob_y_right[:-1], np.logical_not(neighbours_forward_y))

        ind_prev = np.searchsorted(positions[1], y-1)
        if positions[1][ind_prev] == y-1:
            global_flag = global_flag and not second_flag and prob_y_right[ind_prev]
            #print(global_flag)
            mask[ind_prev] = global_flag


        mask[:-1] = np.logical_or(mask[:-1], np.logical_and(neighbours_forward_y, np.logical_and(mask[1:], prob_y_right[:-1])))

        ind_prev = np.searchsorted(positions[1], y-1)
        if positions[1][ind_prev] == y-1:
            global_flag = global_flag and not second_flag and prob_y_right[ind_prev]
            #print(global_flag)
            mask[ind_prev] = global_flag


        mask[positions[1] >= y] = False #Choose the particules on the right side
        """
        ind_prev = np.searchsorted(positions[1], -1)
        if positions[1][ind_prev] == -1:
            global_flag = global_flag and not second_flag and mask[ind_prev]
            mask[ind_prev] = global_flag
        """
        positions[1] += mask #These particles go right

        dist[0] = positions[0][1:] - positions[0][:-1]
        dist[1] = positions[1][1:] - positions[1][:-1]

        return positions, dist


    def _bernoulli_right_parallel_multi(self, positions, dist, **kwargs):
        raise NotImplementedError

    def _bernoulli_both_parallel_multi(self, positions, dist, **kwargs):
        raise NotImplementedError

    def _update_axis_right(self, positions, dist, blocking_mask, id_1, id_2):
        mask = np.zeros(shape = positions.shape[0], dtype=np.bool_)
        mask_2 = np.zeros(shape = positions.shape[0], dtype=np.bool_)

        neighbours_forward = (dist == 1).astype(np.bool_)
        u = (np.random.random(size = positions.shape[0]) < self.p[id_1, id_2]).astype(np.bool_)
        mask[:-1] = np.logical_and(u[:-1], np.logical_not(neighbours_forward))
        mask[:-1] = np.logical_and(mask[:-1], blocking_mask[:-1])
        mask[-1] = u[-1] and  blocking_mask[-1]
        positions[mask] += 1
        mask_2[:-1] = np.logical_and(np.logical_and(u[:-1], neighbours_forward), mask[:-1])
        positions[mask_2] += 1


        mask_f = np.logical_or(mask, mask_2)
        prod =  np.logical_and(mask_f, u).astype(int)
        dist += prod[1:] - prod[:-1]

        return positions, dist
    
    def _update_axis_both(self, positions, dist, blocking_mask, id_1, id_2):
        u = np.random.choice([-1,0,1], size = positions.shape[0], p=[self.q[id_1,id_2], 1 - self.p[id_1,id_2] - self.q[id_1,id_2],self.p[id_1,id_2]])
        
        mask_u = (u == 1).astype(np.bool_)
        mask_u = np.logical_and(mask_u, blocking_mask)
        positions -= np.logical_and(u==-1, blocking_mask)
        while True:
            mask = (positions[1:] == positions[:-1]).astype(np.bool_)
            if not mask.any():
                break
            else:
                positions[:-1][mask] -= 1

        dist = positions[1:] - positions[:-1]
        neighbours = (dist == 1).astype(np.bool_)
        mask = np.zeros(positions.shape[0], dtype=np.bool_)
        mask[-1] = mask_u[-1]
        mask[:-1] = np.logical_and(np.logical_not(neighbours), mask_u[:-1])
        mask[:-1]= np.logical_or(mask[:-1], np.logical_and(mask[1:], mask_u[:-1]))

        positions += mask 
        dist = positions[1:] - positions[:-1]
        
        return positions, dist
        

    def _bernoulli_right_sequential_HD(self, positions, dist, **kwargs):
        x_axis = kwargs["x_axis"]
        y_axis = kwargs["y_axis"]
        n_axis = kwargs["n_axis"]
        order = np.arange(n_axis)
        np.random.shuffle(order)
        
        blocking_mat = np.ones((x_axis.shape[0], y_axis.shape[0]), dtype=np.bool_)

        for axis_number in order:
            id_1 = int(axis_number >= x_axis.shape[0])
            id_2 = axis_number % x_axis.shape[0]
            blocking_mask = np.ones(positions[id_1][id_2].shape[0])
            if id_1 == 0:
                pos = y_axis[np.where(blocking_mat[id_2] == 0)]
                blocking_mask = np.logical_not(np.isin(positions[id_1][id_2], pos))
            else:
                pos = x_axis[np.where(blocking_mat[id_2] == 0)]
                blocking_mask = np.logical_not(np.isin(positions[id_1][id_2], pos))
            positions[id_1][id_2], dist[id_1][id_2] = self._update_axis_right(positions[id_1][id_2], dist[id_1][id_2], blocking_mask, id_1, id_2) 
            arr_block = positions[id_1][id_2][np.isin(positions[id_1][id_2], pos)]
            if id_1 == 0:
                for val in arr_block:
                    p = np.where(y_axis == val)
                    blocking_mat[id_2, p] = 0
            else:
                for val in arr_block:
                    p = np.where(x_axis == val)
                    blocking_mat[id_2, p] = 0

                
        return positions, dist




    def _bernoulli_both_sequential_HD(self, positions, dist, **kwargs):
        x_axis = kwargs["x_axis"]
        y_axis = kwargs["y_axis"]
        n_axis = kwargs["n_axis"]
        order = np.arange(n_axis)
        np.random.shuffle(order)
        
        blocking_mat = np.ones((x_axis.shape[0], y_axis.shape[0]), dtype=np.bool_)

        for axis_number in order:
            id_1 = int(axis_number >= x_axis.shape[0])
            id_2 = axis_number % x_axis.shape[0]
            blocking_mask = np.ones(positions[id_1][id_2].shape[0])
            if id_1 == 0:
                pos = y_axis[np.where(blocking_mat[id_2] == 0)]
                blocking_mask = np.logical_not(np.isin(positions[id_1][id_2], pos))
            else:
                pos = x_axis[np.where(blocking_mat[id_2] == 0)]
                blocking_mask = np.logical_not(np.isin(positions[id_1][id_2], pos))
            positions[id_1][id_2], dist[id_1][id_2] = self._update_axis_both(positions[id_1][id_2], dist[id_1][id_2], blocking_mask, id_1, id_2) 
            arr_block = positions[id_1][id_2][np.isin(positions[id_1][id_2], pos)]
            if id_1 == 0:
                for val in arr_block:
                    p = np.where(y_axis == val)
                    blocking_mat[id_2, p] = 0
            else:
                for val in arr_block:
                    p = np.where(x_axis == val)
                    blocking_mat[id_2, p] = 0

                
        return positions, dist

    def _choose_func(self):
        match self.ley_name:
            case "bernoulli_right_sequential_uni":
                return self._bernoulli_right_sequential_uni
            case "bernoulli_both_sequential_uni":
                return self._bernoulli_both_sequential_uni
            case "bernoulli_right_parallel_uni":
                return self._bernoulli_right_parallel_uni
            case "bernoulli_both_parallel_uni":
                return self._bernoulli_both_parallel_uni
            case "bernoulli_right_sequential_multi":
                return self._bernoulli_right_sequential_multi
            case "bernoulli_both_sequential_multi":
                return self._bernoulli_both_sequential_multi
            case "bernoulli_right_parallel_multi":
                return self._bernoulli_right_parallel_multi
            case "bernoulli_both_parallel_multi":
                return self._bernoulli_both_parallel_multi
            case "bernoulli_right_sequential_HD":
                return self._bernoulli_right_sequential_HD
            case "bernoulli_both_sequential_HD":
                return self._bernoulli_both_sequential_HD
            
    def _move_type(self):
        ret = self.ley_name.split("_")[-2:]
        self.direction = self.ley_name.split("_")[-3]
        return ret
    
    def __call__(self, positions, dist, **kwargs):
        return self.call_func(positions, dist, **kwargs)     #if A and B; A: if the particle has to move (binomial) on the y axis; B: the particle souldn't have a forward neighbour on the y axis
           