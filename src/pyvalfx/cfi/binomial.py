import numpy as np


class BinomialCRR:
    """
    Cox-Ross-Rubinstein binomial lattice model for option pricing with the given parameters.

    Parameters:
    S: The spot price of the underlying asset
    T: The total time horizon from valuation date
    sigma: The volatility of the underlying asset
    r: The risk-free interest rate
    M: The number of time steps
    q: The continuous dividend yield (default is 0)

    Underlying asset price assumed to follow a geometric Brownian motion.

    Generates shape-(M+1xM+1) array lattice where:
    M is the number of metric periods

    """

    def __init__(
        self,
        S: float,
        T: float,
        sigma: float,
        r: float,
        M: int,
        q: float = 0,
    ):
        # if any(x <= 0 for x in [S, T, sigma, r]):
        #     raise ValueError("Expected inputs S, T, sigma, rfr to be greater than 0")

        self.S = S
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q
        self.M = int(M)

        # # Check if non-scalar T has at least as many periods as S
        # if self.T.shape < self.S.shape:
        #     raise ValueError("S and T inputs must have the same dimensions")

    @property
    def dt(self):
        """Length of time for each time step"""
        return self.T / self.M

    @property
    def u(self):
        """Up factor"""
        return np.exp(self.sigma * np.sqrt(self.dt))

    @property
    def d(self):
        """Down factor"""
        return 1 / self.u

    @property
    def p_u(self):
        """Probability of up movement"""
        return (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    @property
    def p_d(self):
        """Probability of down movement"""
        return 1 - self.p_u

    def generate_lattice(self):
        lattice = np.zeros((self.M + 1, self.M + 1))
        lattice[0, 0] = self.S

        for i in range(1, self.M + 1):
            lattice[:i, i] = lattice[:i, i - 1] * self.u
            lattice[i, i] = lattice[i - 1, i - 1] * self.d

        return lattice


# TEMP TESTING EXAMPLE
# z = BinomialCRR(10, 5, 0.45, 0.05, round(5/(1/52)))
# ST = z.generate_lattice()
# payoff = np.zeros(ST.shape)
# payoff[:,-1] = np.maximum(ST[:,-1] - 10, 0)
# for i in reversed(range(1, z.M+1)):
#     payoff[:i, i-1] = (z.p_u*payoff[:i, i] + z.p_d*payoff[1:i+1, i])*np.exp(-z.r*z.dt)
