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
        #return pd.read_csv(path, skiprows=skiprows, delim_whitespace=True)
        return pd.read_csv(path, skiprows=skiprows, sep=r'\s+')

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

    # def _load_data(self):
    #     data = read_csv_with_multichar_comments(self.bkfile, self.comment_prefix)

    #     expected_cols = {'y', 'log10_k', 'log10_S(k)'}
    #     if not expected_cols.issubset(data.columns):
    #         raise ValueError(
    #             f"CSV must contain columns {expected_cols}, but found {list(data.columns)}"
    #         )

    #     # Keep log10_k for building the interpolator — convex hull stays rectangular
    #     self.y       = data['y'].to_numpy()
    #     self.log10_k = data['log10_k'].to_numpy()
    #     self.S       = np.power(10.0, data['log10_S(k)'].to_numpy())  # S in linear scale

    #     self.y_range = (self.y.min(), self.y.max())
    #     self.k_range = (10**self.log10_k.min(), 10**self.log10_k.max())

    def _load_data(self):
        data = read_csv_with_multichar_comments(self.bkfile, self.comment_prefix)

        expected_cols = {'y', 'log10_k', 'log10_S(k)'}
        if not expected_cols.issubset(data.columns):
            raise ValueError(
                f"CSV must contain columns {expected_cols}, but found {list(data.columns)}"
            )


        # Drop rows where log10_S(k) is NaN or -inf (S=0 exactly, unphysical for interpolation)
        bad_mask = data['log10_S(k)'].isna() | np.isinf(data['log10_S(k)'])
        if bad_mask.sum() > 0:
            # print(f"[DEBUG] Dropping {bad_mask.sum()} bad rows (NaN or -inf in log10_S(k))")
            data = data[~bad_mask].reset_index(drop=True)

        #print(f"[DEBUG] Rows after cleaning: {len(data)}")

        self.y       = data['y'].to_numpy()
        self.log10_k = data['log10_k'].to_numpy()
        self.S       = np.power(10.0, data['log10_S(k)'].to_numpy())

        self.y_range = (self.y.min(), self.y.max())
        self.k_range = (10**self.log10_k.min(), 10**self.log10_k.max())

    # def _build_interpolator(self):
    #     # (y, log10_k) space: uniform point distribution, well-conditioned triangulation
    #     points = np.column_stack((self.y, self.log10_k))
    #     self.interpolator = interpolate.CloughTocher2DInterpolator(
    #         points, self.S, rescale=True
    #     )

    def _build_interpolator(self):
        points = np.column_stack((self.y, self.log10_k))

        # ── Check 1: any NaN in S values before building ─────────────────────
        nan_mask = np.isnan(self.S)

        if nan_mask.sum() > 0:
            raise ValueError(
                f"Input data contains {nan_mask.sum()} NaN values in S — "
                f"fix the source data before building interpolator."
            )

        self.interpolator = interpolate.CloughTocher2DInterpolator(
            points, self.S, rescale=True
        )

        # ── Check 2: verify interpolator reproduces a known data point ────────
        test_y  = self.y[len(self.y)//2]
        test_k  = self.log10_k[len(self.log10_k)//2]
        test_S  = self.S[len(self.S)//2]
        result  = self.interpolator(test_y, test_k)

        if np.isnan(result):
            raise RuntimeError(
                "CloughTocher interpolator returns NaN even on its own training data. "
                "This usually means the log10_S(k) column in the file contains "
                "non-finite values (inf/-inf) that survive the power(10, x) conversion. "
                "Check: np.isinf(data['log10_S(k)'].to_numpy()).any()"
            )

    # def S_dipole(self, y_val, k_val):
    #     y_val = np.atleast_1d(np.asarray(y_val, dtype=float))
    #     k_val = np.atleast_1d(np.asarray(k_val, dtype=float))


    #     # # ── Guard: catch zero / negative k before log10 ─────────────────────
    #     if np.any(k_val <= 0):
    #         raise ValueError(
    #             f"k_val must be strictly positive for log10 conversion, got: {k_val}"
    #         )

    #     log10_k_val = np.log10(k_val)

    #     # ── Guard: check query is inside the interpolation domain ───────────
    #     y_in  = (y_val  >= self.y_range[0])      & (y_val  <= self.y_range[1])
    #     k_in  = (log10_k_val >= self.log10_k.min()) & (log10_k_val <= self.log10_k.max())
    #     if not np.all(y_in):
    #         print(f"[WARN] y_val outside data range {self.y_range}: {y_val[~y_in]}")
    #     if not np.all(k_in):
    #         print(f"[WARN] log10(k_val) outside data range "
    #             f"[{self.log10_k.min():.2f}, {self.log10_k.max():.2f}]: "
    #             f"{log10_k_val[~k_in]}")

    #     result = self.interpolator(y_val, log10_k_val)


    #     return result

    def S_dipole(self, y_val, k_val):
        y_val = np.atleast_1d(np.asarray(y_val, dtype=float))
        k_val = np.atleast_1d(np.asarray(k_val, dtype=float))

        # Return NaN for unphysical k (zero or negative) instead of raising
        bad_k = k_val <= 0
        if np.any(bad_k):
            print(f"[WARN] {bad_k.sum()} k_val entries are <= 0, setting S=NaN there")

        log10_k_val = np.where(bad_k, np.nan, np.log10(np.where(bad_k, 1.0, k_val)))

        result = self.interpolator(y_val, log10_k_val)

        # Stamp NaN over the bad-k positions
        result = np.where(bad_k, np.nan, result)

        return result