#!/bin/python3

import math
import os
import random
import re
import sys


# Complete the 'getMinimumDifference' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. STRING_ARRAY a
#  2. STRING_ARRAY b

def getMinimumDifference(a, b):
    result = []
    for a_word, b_word in zip(a, b):
        if len(a_word) != len(b_word):
            result.append(-1)
        else:
            num_changes = 0
            counter = dict()
            for char in a_word:
                counter[char] = 1 if (char not in counter) else (counter[char] + 1)
            for char in b_word:
                if char in counter:
                    if counter[char] == 1:
                        del counter[char]
                    else:
                        counter[char] -= 1
                else:
                    num_changes += 1
            result.append(num_changes)
    return result