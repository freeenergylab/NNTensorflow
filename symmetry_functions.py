#!/usr/bin/env python

from __future__ import print_function
import numpy as np

#===============================================================================
# Symmetry Function 1: Radial Function
#===============================================================================
def RadialFunction(R_s, R_c, eta, prmtop, mdcrd, nframe, QM_index):
    """ Compute the Radial Function Value of Atom i.

    Parameters
    ----------
    R_s : float
         Predetermined parameter in radial function    
    R_c : float
         Predetermined cutoff radius in cutoff function
    eta : dict
         Predetermined parameters in radial function
                       keys => values
               each element => eta value (in unit of 1/A**2)
         Here, the same element type has the same eta value
    prmtop : str
         Topology file name
    mdcrd : str
         Trajectory file name
    nframe : int
         Number of trajectory frames
    QM_index : list
         Index of QM atoms

    Returns
    -------
    G1 : np.ndarray
        The radial function value of atom i in k frames
    """

    import parm_AMBER
    natom, atom_name, atom_charge = parm_AMBER.ReadParm(prmtop = prmtop)
    import mdcrd_AMBER
    coor_kn = mdcrd_AMBER.ReadTraj(mdcrd = mdcrd, nframe = nframe, natom = natom)
    nqm = len(QM_index)
    import distance
    import cutoff_function
    G1 = np.zeros([nframe, nqm], np.float64)
    for k in range(nframe):
        for i in range(nqm):
            for j in range(nqm):
                if(j != i): 
                    R_ij = distance.Distance(vector1 = coor_kn[k, QM_index[j], :], 
                                             vector2 = coor_kn[k, QM_index[i], :])
                    G1[k, i] += np.exp(-eta[atom_name[QM_index[i]]] * (R_ij - R_s)**2.0) \
                              * cutoff_function.CutoffFunction(R_ij = R_ij, R_c = R_c)
    return G1

#===============================================================================
# Symmetry Function 2: Angular Function
#===============================================================================
def AngularFunction(R_c, lmda, eta, ksi, prmtop, mdcrd, nframe, QM_index):
    """ Compute the Angular Function Value of Atom i.

    Parameters
    ----------
    R_c : float
         Predetermined cutoff radius in cutoff function
    lmda : float
         Predetermined parameter (here, 1.0 or -1.0) in angular function
    eta : dict
         Predetermined parameters in angular function
                       keys => values
               each element => eta value (in unit of 1/A**2)
         Here, the same element type has the same eta value
    ksi : dict
         Predetermined parameters in angular function
                       keys => values
               each element => ksi value
         Here, the same element type has the same ksi value
    prmtop : str
         Topology file name
    mdcrd : str
         Trajectory file name
    nframe : int
         Number of trajectory frames
    QM_index : list
         Index of QM atoms

    Returns
    -------
    G2 : np.ndarray
        The angular function value of atom i in k frames
    """

    import parm_AMBER
    natom, atom_name, atom_charge = parm_AMBER.ReadParm(prmtop = prmtop)
    import mdcrd_AMBER
    coor_kn = mdcrd_AMBER.ReadTraj(mdcrd = mdcrd, nframe = nframe, natom = int(natom))
    nqm = len(QM_index)
    import distance
    import cutoff_function
    import cosine_angle_ijk
    G2 = np.zeros([nframe, nqm], np.float64)
    for k in range(nframe):
        for i in range(nqm):
            for j in range(nqm):
                if (j != i): 
                    R_ij = distance.Distance(vector1 = coor_kn[k, QM_index[j], :], 
                                             vector2 = coor_kn[k, QM_index[i], :])
                    for l in range(nqm):
                        if (l != i): 
                            R_il = distance.Distance(vector1 = coor_kn[k, QM_index[l], :], 
                                                     vector2 = coor_kn[k, QM_index[i], :])

                            R_jl = distance.Distance(vector1 = coor_kn[k, QM_index[l], :], 
                                                     vector2 = coor_kn[k, QM_index[j], :])

                            cos_ijk = cosine_angle_ijk.CosineAngle(vector1 = coor_kn[k, QM_index[i], :], 
                                                                   vector2 = coor_kn[k, QM_index[j], :], 
                                                                   vector3 = coor_kn[k, QM_index[l], :])
                            
                            # Avoid that """RuntimeWarning: invalid value encountered in double_scalars"""
                            # such as "FloatingPointError: invalid value encountered in power"
                            if ((round(lmda, 1) == -1.0) and (round(cos_ijk, 1) == 1.0)):
                                G2[k, i] += (2.0**(1.0 - ksi[atom_name[QM_index[i]]])) \
                                          * ((0.0)**ksi[atom_name[QM_index[i]]]) \
                                          * (np.exp(-eta[atom_name[QM_index[i]]] * (R_ij**2.0 + R_jl**2.0 + R_il**2.0))) \
                                          * (cutoff_function.CutoffFunction(R_ij = R_ij, R_c = R_c)) \
                                          * (cutoff_function.CutoffFunction(R_ij = R_jl, R_c = R_c)) \
                                          * (cutoff_function.CutoffFunction(R_ij = R_il, R_c = R_c))
                            else:
                                G2[k, i] += (2.0**(1.0 - ksi[atom_name[QM_index[i]]])) \
                                          * ((1.0 + lmda * cos_ijk)**ksi[atom_name[QM_index[i]]]) \
                                          * (np.exp(-eta[atom_name[QM_index[i]]] * (R_ij**2.0 + R_jl**2.0 + R_il**2.0))) \
                                          * (cutoff_function.CutoffFunction(R_ij = R_ij, R_c = R_c)) \
                                          * (cutoff_function.CutoffFunction(R_ij = R_jl, R_c = R_c)) \
                                          * (cutoff_function.CutoffFunction(R_ij = R_il, R_c = R_c))
    return G2

#===============================================================================
# Use __main__ To Do Test
#===============================================================================
if __name__=='__main__':

    # in unit of 1/A**2 and Bohr_to_A = 0.529177249
    #eta = {'C': 6.428, 'H': 4.285, 'H1': 4.285, 'H2': 4.285, 'Cl': 0.321}
    eta = {'C': 0.714, 'C1': 0.714, 'C2': 0.714, 'C3': 0.714, 'C4': 0.714, 'O': 2.857, 'H': 1.428, 'H1': 1.428, 'H2': 1.428, 'H3': 1.428, 'H4': 1.428, 'H5': 1.428, 'H6': 1.428, 'H7': 1.428}
    # Note: dependent on system
    #QM_index = [0, 1, 2, 3, 4, 5]
    QM_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    G1 = RadialFunction(R_s = 0.0, R_c = 6.0, eta = eta, 
                        prmtop = '../training_data/ligands.parm7', mdcrd = '../training_data/md.crd', nframe = 1000, QM_index = QM_index)
    print(G1)

    #ksi = {'C': 1.80, 'H': 1.20, 'H1': 1.20, 'H2': 1.20, 'Cl': 0.09}
    ksi = {'C': 0.2, 'C1': 0.2, 'C2': 0.2, 'C3': 0.2, 'C4': 0.2, 'O': 0.8, 'H': 0.4, 'H1': 0.4, 'H2': 0.4, 'H3': 0.4, 'H4': 0.4, 'H5': 0.4, 'H6': 0.4, 'H7': 0.4}
    # Note: lmda = 1.0 or -1.0
    lmda = 1.0
    G2 = AngularFunction(R_c = 6.0, lmda = lmda, eta = eta, ksi = ksi, 
                         prmtop = '../training_data/ligands.parm7', mdcrd = '../training_data/md.crd', nframe = 1000, QM_index = QM_index)
    print(G2)

