# Python class file for integrating over Woods-Saxon nuclear density and building interpolator for Woods-Saxon density function T_A(b).

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
import os
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import expit


########-----------------------------------------------------------------------
# Woods-Saxon nuclear density and thickness function T_A(b)
########------------------------------------------------------------------------------  

class WoodsSaxonInterpolator:
    """
    Interpolates the Woods-Saxon nuclear thickness function T_A(b) over impact parameter b.
    """

    def __init__(self, A, RA, d):
        """
        Initialize the interpolator for T_A(b).

        Parameters
        ----------
        A : int
            Mass number of the nucleus.
        R_A : float
            Nuclear radius parameter (in GeV-1).
        d : float
            Surface thickness parameter (in GeV-1).
        n : float
            Normalization constant for the Woods-Saxon density.
        """
        self.A = A
        self.RA = RA
        self.d = d
        self.n = self._compute_normalization()

         
    
        #self._build_interpolator()

    
    def unnormalized_woods_saxon_density(self, z, b):
        """Unnormalized Woods-Saxon nuclear density function ρ_A(z, b)."""
        r = np.sqrt(z**2 + b**2)
        #return 1 / (1 + np.exp((r - self.RA) / self.d)) #was causing problem in cubi method interpolator of b dep dipole
        return expit(-(r - self.RA) / self.d)   # expit(x) = 1/(1+exp(-x)) check!
    
    def _compute_normalization(self):
        """Compute normalization constant n for Woods-Saxon density."""
        integrand = lambda z, b: (2 * np.pi * b) * self.unnormalized_woods_saxon_density(z, b)

        def integrand1(b):   
            return quad(lambda z: integrand(z, b), -np.inf, np.inf)[0]

        integral, _ = quad(lambda b: integrand1(b), 0, np.inf)
        n = integral
        #print("Normalization constant n for Woods-Saxon density:", n)
        return 1/n  
    
    def rho_A(self, z, b):
        """Normalized Woods-Saxon nuclear density function ρ_A(z, b)."""
        return self.n * self.unnormalized_woods_saxon_density(z, b)
    
    def compute_TA_b(self, b):
        """Calculate the nuclear thickness function T_A(b)."""
        integrand = lambda z: self.rho_A(z, b)
        T_A_b, _ = quad(integrand, -np.inf, np.inf)
        return T_A_b
    
    def TA_b(self, b):
        """Wrapper to compute T_A(b)."""
        return self.compute_TA_b(b)
    
    def integrated_TA_b(self):
        integrand = lambda b: b * self.TA_b(b)
        integral, _ = quad(integrand, 30, np.inf) 
	

        return integral
    
    
    
    