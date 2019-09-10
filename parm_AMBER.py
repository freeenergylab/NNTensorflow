#!/usr/bin/env python

from __future__ import print_function
import numpy as np

#===============================================================================
# Read Paramter from Amber-format Topology-file
#===============================================================================
def ReadParm(prmtop):
    """ Read Topology from prmtop File.

    Parameters
    ----------
    prmtop : str
            Topology file name

    Returns
    -------
    natom : int
           Number of atoms
    atom_name : str
           Name of atoms
    atom_charge : float
           Charge of atoms
    """

    atom_name = list()
    atom_charge = list()
    f = open(prmtop, 'r')
    while True:
        line = f.readline()
        if (line[0:14] == "%FLAG POINTERS"):
            line = f.readline()
            line = f.readline()
            natom = int(line[4:8])
        if (line[0:15] == "%FLAG ATOM_NAME"):
            line = f.readline()
            while True:
                line = f.readline()
                if (line[0:12] == "%FLAG CHARGE"):
                    break
                tokens = line.split()
                for i in range(len(tokens)):
                    atom_name.append(tokens[i])
        if (line[0:12] == "%FLAG CHARGE"):
            line = f.readline()
            while True:
                line = f.readline()
                if (line[0:19] == "%FLAG ATOMIC_NUMBER"):
                    break
                tokens = line.split()
                for i in range(len(tokens)):
                    atom_charge.append(float(tokens[i]))
        if (not line):
            break

    return natom, atom_name, atom_charge

#===============================================================================
# Use __main__ To Do Test
#===============================================================================
if __name__=='__main__':
    natom, atom_name, atom_charge = ReadParm(prmtop = '../training_data/ligands.parm7')
    print(natom, atom_name, atom_charge)
    print(type(natom), type(atom_name[0]), type(atom_charge[0]))

