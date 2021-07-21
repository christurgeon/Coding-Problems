#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'findBeforeMatrix' function below.
#
# The function is expected to return a 2D_INTEGER_ARRAY.
# The function accepts 2D_INTEGER_ARRAY after as parameter.
#

def findBeforeMatrix(after):
    m = len(after)
    n = len(after[0])
    before = [[] for _ in range(m)]
    value = 0
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                value = after[0][0]
            elif i == 0:
                value = after[0][j] - after[0][j-1]
            elif j == 0:
                value = after[i][0] - after[i-1][0]
            else:
                value = after[i][j] + after[i-1][j-1] - after[i][j-1] - after[i-1][j]   
            before[i].append(value)
    return before
                
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    after_rows = int(input().strip())
    after_columns = int(input().strip())

    after = []

    for _ in range(after_rows):
        after.append(list(map(int, input().rstrip().split())))

    result = findBeforeMatrix(after)

    fptr.write('\n'.join([' '.join(map(str, x)) for x in result]))
    fptr.write('\n')

    fptr.close()
