from .chaffe import Chaffe


class DifferentialPut:
    """
    Calculates discount for lack of marketability based on the Differential
    European Put Option Model
    """

    def __init__(self, T, sigma_preferred, sigma_common, r, q=0):
        self.T = T
        self.sigma_preferred = sigma_preferred
        self.sigma_common = sigma_common
        self.r = r
        self.q = q

    def calculate_dlom(self):
        dlom_preferred = Chaffe(self.T, self.sigma_preferred, self.r, self.q).calculate_dlom()
        dlom_common = Chaffe(self.T, self.sigma_common, self.r, self.q).calculate_dlom()
        return 1 - (1 - dlom_common) / (1 - dlom_preferred)

    citation = (
        "Ghaidarov, S. '"
        "The Use of Protective Put Options in Quantifying Marketability Discounts Applicable to Common and Preferred"
        " Interests"
        "' Business Valuation Review, Vol 28, No. 2, pg. 88-99"
    )
