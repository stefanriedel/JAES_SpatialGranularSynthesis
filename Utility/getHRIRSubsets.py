import numpy as np
from os.path import dirname, join as pjoin

def getHRIR_ChannelSubset(angular_distribution, hrir_2D, hrir_3D):
    root_dir = dirname(__file__)
    hrir_l_2D = hrir_2D[:,0,:]
    hrir_r_2D = hrir_2D[:,1,:]

    hrir_l_3D = hrir_3D[:,0,:]
    hrir_r_3D = hrir_3D[:,1,:]

    L1_idcs = np.load(pjoin(root_dir, 'L1_idcs_FULL2DEG.npy'))
    L2_idcs = np.load(pjoin(root_dir, 'L2_idcs_FULL2DEG.npy'))
    L3_idcs = np.load(pjoin(root_dir, 'L3_idcs_FULL2DEG.npy'))
    SP_idcs = L1_idcs[[2,10]]
    ZEN_idcs = L3_idcs[4]

    if angular_distribution == 'Uniform_2D':
        hrir_l = hrir_l_2D
        hrir_r = hrir_r_2D
        num_channels = hrir_l.shape[0]
    if angular_distribution == 'L1':
        hrir_l = hrir_l_3D[L1_idcs,:]
        hrir_r = hrir_r_3D[L1_idcs,:]
        num_channels = hrir_l.shape[0]
    if angular_distribution == 'L2':
        hrir_l = hrir_l_3D[L2_idcs,:]
        hrir_r = hrir_r_3D[L2_idcs,:]
        num_channels = hrir_l.shape[0]
    if angular_distribution == 'L3':
        hrir_l = hrir_l_3D[L3_idcs,:]
        hrir_r = hrir_r_3D[L3_idcs,:]
        num_channels = hrir_l.shape[0]
    if angular_distribution == 'L2L3':
        hrir_l = hrir_l_3D[np.concatenate((L2_idcs,L3_idcs)),:]
        hrir_r = hrir_r_3D[np.concatenate((L2_idcs,L3_idcs)),:]
        num_channels = hrir_l.shape[0]
    if angular_distribution == 'ZEN':
        hrir_l = np.array([hrir_l_3D[ZEN_idcs,:]])
        hrir_r = np.array([hrir_r_3D[ZEN_idcs,:]])
        num_channels = hrir_l.shape[0]
    if angular_distribution == 'SP':
        hrir_l = hrir_l_3D[SP_idcs,:]
        hrir_r = hrir_r_3D[SP_idcs,:]
        num_channels = hrir_l.shape[0]

    return hrir_l, hrir_r, num_channels



"""
# Here is some example code to find indices
from Utility.ambisonics import sph2cart

hrir_3D_sph = hrir_3D_dataset[1] # azimuth, elevation in deg.
hrir_3D_azi_ele = hrir_3D_sph[:,:2]
cube_sph_coord = np.array([ [0,0], [-22.5,0], [-45,0], [-75,0], [-105,0], [-135,0], [-180,0], [135,0], [105,0], [75,0], [45,0], [22.5,0], 
            [-22.5,30], [-67.5,30], [-112.5,30], [-157.5,30], [157.5,30], [112.5,30], [67.5,30], [22.5,30], 
            [0,60], [-90,60], [-180,60], [90,60], [0,90]])
cube_xyz = sph2cart(cube_sph_coord[:,0]/180.0*np.pi, (90.0-cube_sph_coord[:,1])/180.0*np.pi).T
hrir_3D_xyz = sph2cart(hrir_3D_azi_ele[:,0] / 180.0 * np.pi, (90.0-hrir_3D_azi_ele[:,1]) / 180.0 * np.pi).T

L1_idcs = np.zeros(12)
for i in range(L1_idcs.size):
    distances = np.zeros(hrir_3D_xyz.shape[0])
    for grid_idx in range(hrir_3D_xyz.shape[0]):
        distances[grid_idx] = np.arccos(np.dot(cube_xyz[i,:], hrir_3D_xyz[grid_idx,:]))
    L1_idcs[i] = int(np.argmin(distances))
L1_idcs = L1_idcs.astype(int)
np.save(pjoin(utility_dir, 'L1_idcs_FULL2DEG.npy'), L1_idcs)

L2_idcs = np.zeros(8)
for i in range(L2_idcs.size):
    distances = np.zeros(hrir_3D_xyz.shape[0])
    for grid_idx in range(hrir_3D_xyz.shape[0]):
        distances[grid_idx] = np.arccos(np.dot(cube_xyz[12+i,:], hrir_3D_xyz[grid_idx,:]))
    L2_idcs[i] = int(np.argmin(distances))
L2_idcs = L2_idcs.astype(int)
np.save(pjoin(utility_dir, 'L2_idcs_FULL2DEG.npy'), L2_idcs)

L3_idcs = np.zeros(5)
for i in range(L3_idcs.size):
    distances = np.zeros(hrir_3D_xyz.shape[0])
    for grid_idx in range(hrir_3D_xyz.shape[0]):
        distances[grid_idx] = np.arccos(np.dot(cube_xyz[12+8+i,:], hrir_3D_xyz[grid_idx,:]))
    L3_idcs[i] = int(np.argmin(distances))
L3_idcs = L3_idcs.astype(int)
np.save(pjoin(utility_dir, 'L3_idcs_FULL2DEG.npy'), L3_idcs)
"""