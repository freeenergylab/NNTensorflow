#!/usr/bin/env python

from __future__ import print_function
import numpy as np

#===============================================================================
# Compute the Cosine of Angle between Three Atoms ijk Centered on Atom i
#===============================================================================
def CosineAngle(vector1, vector2, vector3):
    """ Compute the Cosine of Angle between Three Atoms ijk Centered on Atom i.

    Parameters
    ----------
    vector1 : np.ndarray
             Collection of 3 coordinates for atom i 
    vector2 : np.ndarray
             Collection of 3 coordinates for atom j 
    vector3 : np.ndarray
             Collection of 3 coordinates for atom k 

    Returns
    -------
    cosa : float
          The cosine of angle between three atoms
    """

    v1 = vector1 - vector2
    v2 = vector1 - vector3

    import math
    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))
    
    def length(v):
        return math.sqrt(dotproduct(v, v))
    
    cosa = dotproduct(v1, v2) / (length(v1) * length(v2))

    return cosa
     
#===============================================================================
# Use __main__ To Do Test
#===============================================================================
if __name__=='__main__':

    import mdcrd_AMBER
    nframe = 1000
    natom = 6041
    coor_kn = mdcrd_AMBER.ReadTraj(mdcrd = '../training_data/md.crd', nframe = nframe, natom = natom)
    
    cosa = CosineAngle(vector1 = coor_kn[0, 0, :], 
                       vector2 = coor_kn[0, 1, :], 
                       vector3 = coor_kn[0, 2, :])
    print(cosa)
    import math
    print(math.acos(cosa))
