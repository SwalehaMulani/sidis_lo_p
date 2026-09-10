# Python code for reading and interpolating the dipole amplitude dependent on impact parameter b from BK solver output files in mommentum space.

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
import os

def read_csv_with_multichar_comments(path, comment_prefix="##"):
        """Read a CSV file, skipping lines that start with a multi-character comment prefix."""
        skiprows = 0
        with open(path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(comment_prefix) or stripped == "":
                    skiprows += 1
                else:
                    break
        return pd.read_csv(path, skiprows=skiprows, delim_whitespace=True)

class BKDipoleMomentum:
    """
    Interpolates S(k) values over (b, y, k) using LinearNDInterpolator.
    Automatically converts log10 values from the CSV into normal scale.
    """

    def __init__(self, bk_file, comment_char='##'):
        """
        Initialize the interpolator with data from a CSV file.

        Parameters
        ----------
        bk_file : str
            Path to the CSV file containing columns 'b', 'y', 'k', 'S' in log10() form.
            comment_char : str, optional
            Lines starting with this character will be skipped (default: '##').
        """
        self.bkfile = bk_file
        self.comment_prefix = comment_char
        self._load_data()
        self._build_interpolator()

   

    def _load_data(self):
        """Load bkfile, check for correct columns, and convert from log10 to linear scale."""
        data =read_csv_with_multichar_comments(self.bkfile, self.comment_prefix)

        expected_cols = {'b', 'y', 'log10_k', 'log10_S(k)'}
        if not expected_cols.issubset(data.columns):
            raise ValueError(
                f"CSV must contain columns {expected_cols}, but found {list(data.columns)}"
            )

        # Convert from log10(x) to x (linear) and load into numpy arrays
        self.b = np.sort(np.unique(data['b']))
        self.y = np.sort(np.unique(data['y']))
        self.k = 10**np.sort(np.unique(data['log10_k']))    
        
        # Reshape S into 3D array for RegularGridInterpolator
        # Assumes data is sorted as: b fastest, then y, then k (or adjust as needed)
        S_linear = 10 ** data['log10_S(k)'].to_numpy()
        self.S_grid = S_linear.reshape(len(self.b), len(self.y), len(self.k))

    def _build_interpolator(self):
        """Construct LinearNDInterpolator."""
        #print("Building interpolator...")   # debugging
        self.interpolator = interpolate.RegularGridInterpolator((self.b, self.y, self.k), self.S_grid, bounds_error=False, fill_value=None)
        #print(f"RegularGridInterpolator built with grid shape {self.S_grid.shape}.") # debugging
    

    def S_dipole(self, b_val, y_val, k_val):
        """
        Interpolate S(k) at given (b, y, k).

        Parameters
        ----------
        b_val : float or array-like
            b coordinate(s) (linear scale).
        y_val : float or array-like
            y coordinate(s) (linear scale).
        k_val : float or array-like
            k coordinate(s) (linear scale).

        Returns
        -------
        float or ndarray
            Interpolated S(k) value(s) in linear scale.

        """
        #b_val = np.atleast_1d(b_val)
        #y_val = np.atleast_1d(y_val)
        #k_val = np.atleast_1d(k_val)

        #points = np.column_stack((b_val, y_val, k_val))

        #S_dipole = self.interpolator(points)
        S_dipole = self.interpolator((b_val, y_val, k_val))

        # Return scalar if input was scalar
        #if np.isscalar(b_val) and np.isscalar(y_val) and np.isscalar(k_val):
            #return float(S_dipole)  
              
        return S_dipole
    
   #def __call__(self, y_val, k_val):
       #"""Allow the object to be called directly like a function."""
       #return self.interpolate(y_val, k_val)

    