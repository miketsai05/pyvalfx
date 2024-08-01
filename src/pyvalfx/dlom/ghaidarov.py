from statistics import NormalDist

import numpy as np


class Ghaidarov:
    """
    Calculates discount for lack of marketability based on the Ghaidarov
    Average-Strike Put Option Model
    """

    def __init__(self, T, sigma, q=0):
        self.T = T
        self.sigma = sigma
        self.q = q

    @property
    def s2_t(self):
        return self.sigma**2 * self.T

    @property
    def v_root_t(self):
        return np.sqrt(np.log(2 * (np.exp(self.s2_t) - self.s2_t - 1)) - 2 * np.log(self.s2_t))

    def calculate_dlom(self):
        """
        Calculate discount for lack of marketability
        """
        return np.exp(-self.q * self.T) * (2 * NormalDist().cdf(self.v_root_t / 2) - 1)

    def intermediate_calculations(self):
        """
        Calculate intermediate values for the model
        """
        return {"s2_t": self.s2_t, "v_root_t": self.v_root_t}

    citation = (
        "Ghaidarov, S. (2009) 'Analysis and Critique of the Average Strike Put Option Marketability Discount Model', "
        "SSRN, Sep 2009."
    )
