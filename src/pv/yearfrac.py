"""
Module for performing date and present value related calculations for valuation
models
"""

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

def yearfrac(start_date: pd.Timestamp, end_date: pd.Timestamp, basis=0):
    """Calculates time between two dates.

    Replicates Excel yearfrac functionality.

    Basis:
    0: US (NASD) 30/360
    1: Actual/Actual
    2: Actual/360
    3: Actual/365
    4: European 30/360
    5: Actual/365.25 (not an actual option in Excel formula but commonly used)
    """

    # Ensure start date before end date
    if start_date > end_date:
        start_date, end_date = end_date, start_date
        print("Start date after end date!")

    # Calculate numerator
    if basis == 0:  # US (NASD) 30/360
        start_day = start_date.day
        end_day = end_date.day
        if (
            start_date.month == 2
            and start_date.is_month_end
            and end_date.month == 2
            and end_date.is_month_end
        ):
            end_day = 30
        if start_date.month == 2 and start_date.is_month_end:
            start_day = 30
        if (start_day == 30 or start_day == 31) and end_day == 31:
            end_day = 30
        if start_day == 31:
            start_day = 30
        numerator = (
            360 * (end_date.year - start_date.year)
            + 30 * (end_date.month - start_date.month)
            + (end_day - start_day)
        )

    elif basis == 4:  # EURO 30/360
        start_day = 30 if start_date.day == 31 else start_date.day
        end_day = 30 if end_date.day == 31 else end_date.day
        numerator = (
            360 * (end_date.year - start_date.year)
            + 30 * (end_date.month - start_date.month)
            + (end_day - start_day)
        )

    else:  # Actual
        numerator = (end_date - start_date).days

    # Calculate denominator
    if basis in [0, 2, 4]:
        denominator = 360
    elif basis == 3:
        denominator = 365
    elif basis == 5:
        denominator = 365.25
    else:
        days_list = []
        tmpdate = pd.Timestamp(start_date.year, 1, 1)
        while tmpdate.year <= end_date.year:
            days_list.append(366 if tmpdate.is_leap_year else 365)
            tmpdate = pd.Timestamp(tmpdate.year + 1, 1, 1)
        denominator = sum(days_list) / len(days_list)

    return numerator / denominator