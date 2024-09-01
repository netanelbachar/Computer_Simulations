# The python pair style provides a way to define pairwise additive potential
# functions as python script code that is loaded into LAMMPS from a python file
# which must contain specific python class definitions.

# LAMMPS INPUT

# pair_coeff * * pot_lmp.CoulPot D

from __future__ import print_function

import math


class LAMMPSPairPotential(object):
    def __init__(self):
        self.pmap = dict()
        self.units = 'electron'

    def map_coeff(self, name, ltype):
        self.pmap[ltype]=name

    def check_units(self,units):
        if (units != self.units):
           raise Exception("Conflicting units: %s vs. %s" % (self.units, units))


class CoulPot(LAMMPSPairPotential):
    def __init__(self):
        super(CoulPot, self).__init__()
        # set coeffs: D / r_ij
        self.units = 'electron'
        # Set Coefficients
        self.coeff = {'D'  : {'D'  : (31.746329790939217)}}

    def compute_force(self, rsq, itype, jtype):
        coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]
        rinv = 1.0 / rsq
        dialectric = coeff[0]
        return (dialectric*rinv)  # D/r_ij^2

    def compute_energy(self, rsq, itype, jtype):
        coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]
        rinv = 1.0 / math.sqrt(rsq)
        dialectric = coeff[0]
        return (dialectric * rinv)

class DipPot(LAMMPSPairPotential):
    def __init__(self):
        super(DipPot, self).__init__()
        # set coeffs: D
        self.units = 'electron'
        # Set Coefficients
        self.coeff = {'D'  : {'D'  : (31.746329790939217)}}

    def compute_force(self, rsq, itype, jtype):
        coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]
        rinv = 1.0 / math.sqrt(rsq)
        dialectric = coeff[0]
        return (dialectric*rinv*rinv*rinv*rinv*rinv)  # D/r_ij^5

    def compute_energy(self, rsq, itype, jtype):
        coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]
        rinv = 1.0 / math.sqrt(rsq)   # 1/r
        dialectric = coeff[0]
        return (dialectric * rinv*rinv*rinv)  # D/r_ij^3









