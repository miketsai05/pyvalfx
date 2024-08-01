from ..cfi.blackscholes import BlackScholes


class Chaffe:
    """
    Calculates discount for lack of marketability based on the Chaffe
    European Put Option Model
    """

    def __init__(self, T, sigma, r, q=0):
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q

    def calculate_dlom(self):
        return BlackScholes(1, 1, self.T, self.sigma, self.r, self.q).put_price()

    citation = (
        "Chaffe, D.B. (1993) "
        "'Option Pricing as a Proxy for Discount for Lack of Marketability in Private Company Valuations', "
        "Business Valuation Review, 12(4), pp. 182-188."
    )
