#!/usr/bin/env python

from __future__ import print_function
import numpy as np

#===============================================================================
# Cutoff Function
#===============================================================================
def CutoffFunction(R_ij, R_c):
    """ Compute the Value of Cutoff Function between Atom i and Atom j.

    Parameters
    ----------
    R_ij : float
          Distance between atom i and atom j
    R_c : float
         Predetermined cutoff radius

    Returns
    -------
    f : float
       The value of cutoff function between atom i and atom j
    """

    if (R_c == 0.0):
        print('Error Warning: R_c must not be zero!')

    if (R_ij <= R_c): 
        f = 0.5 * (np.cos(np.pi * R_ij / R_c) + 1.0)
    elif (R_ij > R_c):
        f = 0.0

    return f

#===============================================================================
# Use __main__ To Do Test
#===============================================================================
if __name__=='__main__':
    f = CutoffFunction(R_ij = 3.0, R_c = 6.0)
    print(f)
    print(isinstance(f, float))

