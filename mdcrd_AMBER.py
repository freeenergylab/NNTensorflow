#!/usr/bin/env python

from __future__ import print_function
import numpy as np

#===============================================================================
# Read Trajectory from Amber-format mdcrd-file
#===============================================================================
def ReadTraj(mdcrd, nframe, natom):
    """ Read Trajectory from mdcrd File.

    Parameters
    ----------
    mdcrd : str
           Trajectory file name
    nframe : int
           Number of trajectory frames
    natom : int
           Number of atoms

    Returns
    -------
    coor_kn : float
             The coordinates of atoms
    """

    print ("Reading Trajectory ...")
    coor_kn = np.zeros([nframe, natom, 3], np.float64)
    atom_crd = list()
    f = open(mdcrd, 'r')
    line = f.readline()

    while True:
        line = f.readline()
        tokens = line.split()
        for i in range(len(tokens)):
            atom_crd.append(tokens[i])
        if (not line):
            break

    m = 0
    for i in range(nframe):
        for j in range(natom): 
            coor_kn[i, j, 0] = atom_crd[m]
            coor_kn[i, j, 1] = atom_crd[m+1]
            coor_kn[i, j, 2] = atom_crd[m+2]
            m += 3

    return coor_kn

#===============================================================================
# Use __main__ To Do Test
#===============================================================================
if __name__=='__main__':
    nframe = 1000
    natom = 6041
    coor_kn = ReadTraj(mdcrd = '../training_data/md.crd', nframe = nframe, natom = natom)
    print(coor_kn)
    if isinstance(coor_kn, np.ndarray):
        print("coor_kn is a np.ndarray")
    print(coor_kn[0, 0, :])
    print(coor_kn[0, 0, 0], 
          coor_kn[0, 0, 1], 
          coor_kn[0, 0, 2])
    print(coor_kn[nframe-1, natom-1, 0], 
          coor_kn[nframe-1, natom-1, 1], 
          coor_kn[nframe-1, natom-1, 2])
    
