#!/usr/bin/env python

from __future__ import print_function
import numpy as np

#===============================================================================
# Compute the Cartesian Distance between Two Atoms. 
# Ignores Periodic Boundary Conditions.
#===============================================================================
def Distance(vector1, vector2):
    """ Compute the Distance between Atom i and Atom j.

    Parameters
    ----------
    vector1 : np.ndarray
             Collection of 3 coordinates for atom i 
    vector2 : np.ndarray
             Collection of 3 coordinates for atom j 

    Returns
    -------
    d : float
       The distance between atom i and atom j
    """

    d = np.sqrt(np.sum(np.square(vector1 - vector2)))

    return d

#===============================================================================
# Use __main__ To Do Test
#===============================================================================
if __name__=='__main__':

    import mdcrd_AMBER
    nframe = 1000
    natom = 6041
    coor_kn = mdcrd_AMBER.ReadTraj(mdcrd = '../training_data/md.crd', nframe = nframe, natom = natom)
    
    d = Distance(vector1 = coor_kn[0, 0, :], 
                 vector2 = coor_kn[0, 1, :])
    print(d)
