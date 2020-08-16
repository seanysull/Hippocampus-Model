import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time
import random

ROW,COL = 50,50
class Grid():
    def __init__(self):

        self.mm = ROW
        self.nn = COL
        self.TAO = 0.9
        self.II = 0.3
        self.SIGMA = 0.24
        self.SIGMA2 = self.SIGMA ** 2
        self.TT = 0.05
        self.grid_gain = [0.03, 0.05, 0.07, 0.09]
        # self.grid_gain = [0.02, 0.1]
        # self.grid_gain = [0.08]
        self.grid_layers = len(self.grid_gain)
        self.grid_activity = np.random.uniform(0, 1, (self.mm, self.nn, self.grid_layers))
        # self.distTri = self.buildTopology_s(self.mm, self.nn)
        self.distTri = self.buildTopology(self.mm, self.nn)
        # pass

       
        
    def buildTopology(self, mm, nn):  # Build connectivity matrix     ### Eq 4
        # build x and y positions of cells
        mmm = (np.arange(mm) + (0.5 / mm)) / mm
        nnn = ((np.arange(nn) + (0.5 / nn)) / nn) * np.sqrt(3) / 2
        xx, yy = np.meshgrid(mmm, nnn)
        # ravel the cell position matrices so we can build a cell connectivity matrix
        xx_expanded, yy_expanded = np.meshgrid(xx.ravel(), yy.ravel())
        xy = np.stack([xx_expanded, yy_expanded.T], axis=-1)
        xyt = np.stack([xx_expanded.T, yy_expanded], axis=-1)
        Sdist = [(0,0), (-0.5, np.sqrt(3) / 2), (-0.5, -np.sqrt(3) / 2), 
                 (0.5, np.sqrt(3) / 2), (0.5, -np.sqrt(3) / 2), 
                 (-1, 0), (1, 0)]
        distmat = xy - xyt
        # distmat_complex = distmat_s[:,:,0]+distmat_s[:,:,1]*1j
        
        for ii in range(len(Sdist)):
            aaa1_s = np.linalg.norm(distmat, axis=2)
            rrr_s = xy - xyt + Sdist[ii]
            aaa2_s = np.linalg.norm(rrr_s, axis=2)
            iii_s = np.where(aaa2_s < aaa1_s)
            distmat[iii_s] = rrr_s[iii_s]  
        distmat_transposed = np.swapaxes(distmat,0,1)
        # use triangular distance to induce twisted torus shape
        # triangular_distances = np.stack([distmat + Sdist[i] for i in range(len(Sdist))],axis=-1)
        # normed_tri_distances = np.linalg.norm(triangular_distances, axis=2)
        # final_tri_distance = np.min(normed_tri_distances, axis=-1)
        # min_tri_distance_idx = np.argmin(normed_tri_distances, axis=-1)
        # attempt at vectorisation
        # indexer = np.expand_dims(np.argmin(normed_tri_distances, axis=-1), axis=2)
        # mins_vec = np.take_along_axis(triangular_distances, indexer, axis=2)        
        # min_distances = []
        # for ind, val in np.ndenumerate(min_tri_distance_idx):
        #     dist = triangular_distances[ind[0],ind[1],:,val]
        #     min_distances.append(dist)
        
        # min_distances_reshaped = np.array(min_distances).reshape((mm**2,nn**2,2))
        # sanity_check = np.linalg.norm(min_distances_reshaped, axis=2)==final_tri_distance
        
        # distmat_comp = distmat_transposed[:,:,0]+distmat_transposed[:,:,1]*1j 
        return distmat_transposed

    def update(self, speedVector):

        self.speedVector = speedVector
        
        grid_ActTemp = []
        for jj in range(0,self.grid_layers):
            rrr = self.grid_gain[jj]*np.exp(1j*0)
            matWeights = self.updateWeight(self.distTri,rrr)
            activityVect = np.ravel(self.grid_activity[:,:,jj])
            activityVect = self.Bfunc(activityVect, matWeights)
            activityTemp = activityVect.reshape(self.mm,self.nn)
            activityTemp += self.TAO *( activityTemp/np.mean(activityTemp) - activityTemp)
            activityTemp[activityTemp<0] = 0

            self.grid_activity[:,:,jj] = (activityTemp-np.min(activityTemp))/(  np.max(activityTemp)-np.min(activityTemp)) * 30  ##Eq 2
            
    def update_s(self, speedVector):

        self.speedVector = speedVector

        grid_ActTemp = []
        for jj in range(0, self.grid_layers):
            gain = self.grid_gain[jj]
            matWeights = self.updateWeight_s(self.distTri, gain, bias=0)
            activityVect = np.ravel(self.grid_activity[:, :, jj])
            activityVect = self.Bfunc(activityVect, matWeights)
            activityTemp = activityVect.reshape(self.mm, self.nn)
            floating_normalisation = self.TAO * (activityTemp / np.mean(activityTemp) - activityTemp)
            act_normalised = activityTemp + floating_normalisation
            act_normalised[act_normalised < 0] = 0
            grid_activity = (act_normalised - np.min(act_normalised)) / (np.max(act_normalised)
                                                                         - np.min(act_normalised)) ## * 30  ##Eq 2 
            self.grid_activity[:, :, jj] = grid_activity  
            
    def updateWeight(self,topology,rrr): # Slight update on weights based on speed vector.
        matWeights = self.II * np.exp((-abs(topology-rrr*self.speedVector)**2)/self.SIGMA2) - self.TT   ## Eq 3
        return matWeights

    def updateWeight_s(self, topology, gain, bias):  # Slight update on weights based on speed vector.
        rotation_matrix = np.array([[np.cos(bias), -np.sin(bias)],
                                    [np.sin(bias), np.cos(bias)]])
        R_beta = np.dot(rotation_matrix, self.speedVector)
        scaled_velocity = np.multiply(gain, R_beta)
        euclid_norm_tridist = np.linalg.norm(topology-scaled_velocity, axis=2)
        # euclid_norm_tridist = np.linalg.norm(topology-gain*self.speedVector, axis=2)        
        matWeights = self.II * np.exp(np.divide(-np.square(euclid_norm_tridist), self.SIGMA2)) - self.TT  ## Eq 3 
        return matWeights

    def Bfunc(self, activity, matWeights):  ## Eq 1
        delta = np.dot(activity, matWeights)
        activity += delta
        return activity
    
arena_size = 50

arenaX = [0, arena_size]
arenaY = [0, arena_size]

## Initial position
Txx = [arenaX[1] / 2]
Tyy = [arenaY[1] / 2]


def conv(ang):
    x = np.cos(np.radians(ang))
    y = np.sin(np.radians(ang))
    return x, y


def random_navigation(length):
    thetaList = []

    theta = 90
    counter = 0
    lenght_counter = 0
    for i in range(length):
        lenght_counter += 1

        prevTheta = np.copy(theta)

        if (Txx[-1] < 2): theta = np.random.randint(-85, 85)

        if (Txx[-1] > arena_size - 2): theta = np.random.randint(95, 260)

        if (Tyy[-1] < 2): theta = np.random.randint(10, 170)

        if (Tyy[-1] > arena_size - 2): theta = np.random.randint(190, 350)

        Txx.append(Txx[-1] + conv(theta)[0] + np.random.uniform(-0.5, 0.5))
        Tyy.append(Tyy[-1] + conv(theta)[1] + np.random.uniform(-0.5, 0.5))

        cx = abs(Txx[-1] - Txx[-2])
        cy = abs(Tyy[-1] - Tyy[-2])
        h = np.sqrt(cx ** 2 + cy ** 2)
        counter += h

        if (theta != prevTheta or i == length - 1):
            thetaList.append([prevTheta, conv(prevTheta)[0], conv(prevTheta)[1], counter])
            counter = 0

    # plt.plot(Txx, Tyy, '-')
    # plt.show()


random_navigation(50)

Txx = np.array(Txx)
Tyy = np.array(Tyy)

grid = Grid()
print("grid created")
log_grid_cells = []
log_matrices = []
start_time = time.time()
for i in range(1, Txx.size):
    speedVector = np.array([Txx[i] - Txx[i - 1], Tyy[i] - Tyy[i - 1]])
    print ("Speed Vector --->  ", speedVector)
    # speedVector = speedVector.reshape((2,))
    # speedVector = (Txx[i] - Txx[i - 1])+1j*(Tyy[i] - Tyy[i - 1])
    update_start = time.time()
    grid.update_s(speedVector)
    update_done = time.time()-update_start
    print("update time ---->", update_done)
    # grid.update(speedVector)
    activity_flat = grid.grid_activity.flatten()
    activity = np.copy(np.squeeze(grid.grid_activity))
    log_grid_cells.append(activity_flat)
    log_matrices.append(activity)
    a=1
finish_time = time.time() - start_time
print("activity calculated in  "+str(finish_time)+" seconds")
log_grid_cells = np.array(log_grid_cells)
log_matrices = np.array(log_matrices)

xx = np.copy(Txx[1:])
yy = np.copy(Tyy[1:])
# dv_levels = ROW*COL
dv_levels = 1


# =============================================================================
# plt.figure(figsize=(24, 20))
# =============================================================================
# plt.plot(xx, yy)
# plt.figure()

for cell_num in range(0,dv_levels):
# for cell_num in range(60,70):
    # cols = [0,5,8,18,24,32,40,48,56,64,72,80,89]
    cols = list(range(0,90,4))
    # celula = log_grid_cells[:, cell_num*3]
    celula = log_grid_cells[:, cols[cell_num]]

    pos_spike_idx = np.where(celula > celula.max() * .9)[0]
    print(len(pos_spike_idx))
    # whe = np.where(log_grid_cells.max(axis=1) * .9)
    # pos_spike_idx = np.where(log_grid_cells.max(axis=1) > .max() * .9)[0]
    # plt.figure()
    # plt.subplot(3, 3, cell_num + 1)
    color = ["or","og","oc","om","oy","ok","+r","+g","+c","+m","+y","+k",
              "^r","^g","^c","^m","^y","^k","*r","*g","*c","*m","*y","*k"]
    # color = ["or","og","oc","om","oy","ok","+r","+g","+c","+m","+y","+k"]    
    plt.plot(xx[pos_spike_idx], yy[pos_spike_idx], color[cell_num])
    # plt.plot(xx[pos_spike_idx], yy[pos_spike_idx], color[cell_num])
