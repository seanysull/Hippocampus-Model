# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:19:29 2020

@author: seano
"""

import numpy as np
mm=3
nn=3
# mmm = (np.arange(mm) + (1 / mm)) / mm
# nnn = (np.arange(nn) + (5 / nn)) / nn
mmm = np.arange(mm)
nnn = np.arange(nn) 
xx, yy = np.meshgrid(mmm, nnn)
xy = np.stack([xx, yy], axis=-1)
xyt = xy.transpose((1,0,2))
Sdist = [(-0.5, np.sqrt(3) / 2), (-0.5, -np.sqrt(3) / 2)]
distmat = xy - xyt
triangular_distances = np.stack([distmat + Sdist[i] for i in range(len(Sdist))],axis=-1)
td = triangular_distances.ravel()
normed_tri_distances = np.linalg.norm(triangular_distances, axis=2)
final_tri_distance = np.min(normed_tri_distances, axis=-1)
min_tri_distance_idx = np.argmin(normed_tri_distances, axis=-1)
# mtdi = np.broadcast_to(min_tri_distance_idx, (3,3,2))
b = np.repeat(min_tri_distance_idx[:, :, np.newaxis], 2, axis=2)
min_distances = triangular_distances[b,]
mindexes = []
distances = []
for ind, val in np.ndenumerate(min_tri_distance_idx):
    dist = triangular_distances[ind[0],ind[1],:,val]
    distances.append(dist)

testy = np.array(distances).reshape((3,3,2))
proof = np.linalg.norm(testy, axis=2)
a=1