import numpy as np
from os.path import dirname, join as pjoin


def getLoudspeaker_ChannelSubset(angular_distribution):
    cube_sph_coord = np.array([[0, 0], [-22.5, 0], [-45, 0], [-75, 0],
                               [-105, 0], [-135, 0], [-180, 0], [135, 0],
                               [105, 0], [75, 0], [45, 0], [22.5, 0],
                               [-22.5, 30], [-67.5, 30], [-112.5, 30],
                               [-157.5, 30], [157.5, 30], [112.5, 30],
                               [67.5, 30], [22.5, 30], [0, 60], [-90, 60],
                               [-180, 60], [90, 60], [0, 90]])

    L1 = cube_sph_coord[:12, :]
    L2 = cube_sph_coord[12:(12 + 8), :]
    L3 = cube_sph_coord[(12 + 8):(12 + 8 + 5), :]

    if angular_distribution == 'L1':
        azi_ele = L1
        num_channels = azi_ele.shape[0]
    if angular_distribution == 'L1L2':
        azi_ele = np.concatenate((L1, L2))
        num_channels = azi_ele.shape[0]
    if angular_distribution == 'L2':
        azi_ele = L2
        num_channels = azi_ele.shape[0]
    if angular_distribution == 'L3':
        azi_ele = L3
        num_channels = azi_ele.shape[0]
    if angular_distribution == 'L2L3':
        azi_ele = np.concatenate((L2, L3))
        num_channels = azi_ele.shape[0]
    if angular_distribution == 'L1L2L3':
        azi_ele = np.concatenate((L1, L2, L3))
        num_channels = azi_ele.shape[0]
    if angular_distribution == 'ZEN':
        azi_ele = cube_sph_coord[-1, :]
        num_channels = azi_ele.shape[0]
    if angular_distribution == 'SP':
        azi_ele = cube_sph_coord[[2, 10], :]
        num_channels = azi_ele.shape[0]

    return azi_ele, num_channels


def getHRIR_ChannelSubset(angular_distribution, hrir_2D, hrir_3D):
    root_dir = dirname(__file__)

    L1_idcs = np.load(pjoin(root_dir, 'L1_idcs_FULL2DEG.npy'))
    L2_idcs = np.load(pjoin(root_dir, 'L2_idcs_FULL2DEG.npy'))
    L3_idcs = np.load(pjoin(root_dir, 'L3_idcs_FULL2DEG.npy'))
    SP_idcs = L1_idcs[[2, 10]]
    ZEN_idcs = L3_idcs[4]

    if angular_distribution == 'Uniform_2D':
        hrir = hrir_2D
        num_channels = hrir.shape[0]
    if angular_distribution == 'L1':
        hrir = hrir_3D[L1_idcs, ...]
        num_channels = hrir.shape[0]
    if angular_distribution == 'L1L2':
        hrir = hrir_3D[np.concatenate((L1_idcs, L2_idcs)), ...]
        num_channels = hrir.shape[0]
    if angular_distribution == 'L2':
        hrir = hrir_3D[L2_idcs, ...]
        num_channels = hrir.shape[0]
    if angular_distribution == 'L3':
        hrir = hrir_3D[L3_idcs, ...]
        num_channels = hrir.shape[0]
    if angular_distribution == 'L2L3':
        hrir = hrir_3D[np.concatenate((L2_idcs, L3_idcs)), ...]
        num_channels = hrir.shape[0]
    if angular_distribution == 'L1L2L3':
        hrir = hrir_3D[np.concatenate((L1_idcs, L2_idcs, L3_idcs)), ...]
        num_channels = hrir.shape[0]
    if angular_distribution == 'ZEN':
        hrir = np.array([hrir_3D[ZEN_idcs, ...]])
        num_channels = hrir.shape[0]
    if angular_distribution == 'SP':
        hrir = hrir_3D[SP_idcs, ...]
        num_channels = hrir.shape[0]

    return hrir, num_channels


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