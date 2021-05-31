#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'calculateCost' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY arr as parameter.
#

def calculateCost(arr):
    minimum_value = 0
    while len(arr) >= 2:
        minimum = 9999999
        index = -1
        for i in range(1, len(arr)):
            subp = arr[i] * arr[i-1]
            if subp < minimum:
                minimum = subp # dp soln
                index = i
        minimum_value += minimum
        middle = max(arr[index-1], arr[index])
        arr = arr[:index-1] + [middle] + arr[index+1:] # resize
    return minimum_value


if __name__ == '__main__':