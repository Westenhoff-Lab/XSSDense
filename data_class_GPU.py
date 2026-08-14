"""
Class that contains scattering data (GPU version)
"""

import cupy as cp
import pandas as pd


class ScatterData:
    def __init__(self, file=None, sep=','):
        if file is None:
            self.q = cp.zeros(1)
            self.i = cp.zeros(1)
            self.e = []
        else:
            self.parse(file, sep)

    def __str__(self):
        return (
            f"Scatter intensity with {self.i.shape[0]} points\n"
            f"q-range from {cp.nanmin(self.q)} to {self.q[-1]}"
        )

    # ----------------------------------------------------------------------
    def parse(self, file, delim):
        """
        Parses scattering data and stores it as CuPy arrays.
        """
        df = pd.read_csv(
            file,
            names=["q", "I", "e"],
            index_col=False,
            delimiter=delim
        )

        # Convert to CuPy
        self.q = cp.asarray(df["q"].to_numpy())
        self.i = cp.asarray(df["I"].to_numpy())

        # If error column present and not all NaN → use it, else ones
        if "e" in df.columns and not df["e"].isna().all():
            self.e = cp.asarray(df["e"].to_numpy())
        else:
            self.e = cp.ones_like(self.i)

    # ----------------------------------------------------------------------
    def set_data(self, q, i, e=None):
        if e is None:
            e = []
        self.q = cp.asarray(q)
        self.i = cp.asarray(i)
        self.e = cp.asarray(e) if isinstance(e, (list, tuple, pd.Series)) else e
        return

    # ----------------------------------------------------------------------
    def cut_q(self, q_min, q_max):
        """
        Returns a new ScatterData object filtered to q-range.
        """
        indices = cp.logical_and(q_min <= self.q, self.q <= q_max)

        q = self.q[indices]
        i = self.i[indices]
        e = self.e[indices] if self.is_error() else []

        new = ScatterData()
        new.set_data(q, i, e)
        return new

    # ----------------------------------------------------------------------
    def scale_intensity(self, k):
        new = self.copy()
        new.set_data(self.q, k * self.i, self.e)
        return new

    # ----------------------------------------------------------------------
    def scale_q(self, k):
        new = self.copy()
        new.set_data(k * self.q, self.i, self.e)
        return new

    # ----------------------------------------------------------------------
    def is_error(self):
        """Returns True if error array exists and has correct shape."""
        if isinstance(self.e, list):
            return False
        return self.e.shape == self.i.shape

    # ----------------------------------------------------------------------
    def copy(self):
        return ScatterData()

    # ----------------------------------------------------------------------
    def remove_nan(self):
        """Remove NaNs from intensity and associated q + error."""
        indices = cp.where(cp.isnan(self.i) == False)

        q = self.q[indices]
        i = self.i[indices]
        e = self.e[indices] if self.is_error() else []

        new = self.copy()
        new.set_data(q, i, e)
        return new

    # ----------------------------------------------------------------------
    def normalize(self, q_range=False):
        new = self.copy()
        new.q = self.q

        if q_range:
            mask = cp.logical_and(self.q > q_range[0], self.q < q_range[1])
            new.i = self.i / cp.nanmean(self.i[mask])
        else:
            new.i = self.i / cp.nansum(self.i)

        new.e = new.i
        return new

