# Python code for reading and interpolating the dipole amplitude from BK solver output files in mommentum space.

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
    Interpolates S(k) values over (y, k) using CloughTocher2DInterpolator.
    Automatically converts log10 values from the CSV into normal scale.
    """

    def __init__(self, bk_file, comment_char='##'):
        """
        Initialize the interpolator with data from a CSV file.

        Parameters
        ----------
        bk_file : str
            Path to the CSV file containing columns 'y', 'k', 'S' in log10() form.
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

        expected_cols = {'y', 'log10_k', 'log10_S(k)'}
        if not expected_cols.issubset(data.columns):
            raise ValueError(
                f"CSV must contain columns {expected_cols}, but found {list(data.columns)}"
            )

        # Convert from log10(x) to x (linear) and load into numpy arrays
        self.y = data['y'].to_numpy()
        self.k = np.power(10, data['log10_k'].to_numpy())
        self.S = np.power(10, data['log10_S(k)'].to_numpy())

    def _build_interpolator(self):
        """Construct CloughTocher2DInterpolator."""
        points = np.column_stack((self.y, self.k))
        self.interpolator = interpolate.CloughTocher2DInterpolator(points, self.S)

    def S_dipole(self, y_val, k_val):
        """
        Interpolate S(k) at given (y, k).

        Parameters
        ----------
        y_val : float or array-like
            y coordinate(s) (linear scale).
        k_val : float or array-like
            k coordinate(s) (linear scale).

        Returns
        -------
        float or ndarray
            Interpolated S(k) value(s) in linear scale.

        """

        #k_min, k_max = self.k_range
        #Y_min, Y_max = self.Y_range
        
        #if k_val < 0 or y_val < 0:
            #return np.nan
        
        #if y_val > Y_max:
            #return np.nan
        
        #if k_val > k_max and 0 <= y_val <= Y_max: # large k, Y within limits
            #return 1.0 
        
        #if  k_val < k_min and 0 <= y_val <= Y_max: # small k, Y within limits
            #return 0.0
        S_dipole = self.interpolator(y_val, k_val)
        
        return S_dipole
   #def __call__(self, y_val, k_val):
       #"""Allow the object to be called directly like a function."""
       #return self.interpolate(y_val, k_val)

    