
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'predictMissingPrices' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. STRING startDate
#  2. STRING endDate
#  3. STRING_ARRAY knownTimestamps
#  4. INTEGER_ARRAY pricesAtKnownTimeStamps
#  5. STRING_ARRAY unknownTimestamps
#

from collections import OrderedDict
import numpy as np
import pandas as pd

def predictMissingPrices(startDate, endDate, knownTimestamps, pricesAtKnownTimeStamps, unknownTimestamps):
    all_prices = []
    for i, tstamp in enumerate(knownTimestamps):
        all_prices.append( (tstamp, pricesAtKnownTimeStamps[i]) )
    for i in unknownTimestamps:
        all_prices.append( (i, None) )
    all_prices = sorted(all_prices, key = lambda x : x[0])
        
    prices_missing_indices = set([i for i in range(len(all_prices)) if all_prices[i][1] is None])

    prices_refined = []
    for _, price in all_prices:
        prices_refined.append(np.nan if price is None else price)
    print(prices_refined)
    s = pd.Series(prices_refined)
    s = s.interpolate()
    
    revised = [val for idx, val in enumerate(s) if (idx in prices_missing_indices)]
    
    return revised
        
