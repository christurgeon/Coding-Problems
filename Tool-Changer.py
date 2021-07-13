#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'toolchanger' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING_ARRAY tools
#  2. INTEGER startIndex
#  3. STRING target
#


def toolchanger(tools, startIndex, target):
    if tools[startIndex] == target:
        return startIndex
    size = len(tools)
    l, r = startIndex-1, startIndex+1
    while (l % size) != startIndex:
        if tools[l % size] == target:
            return abs(l - startIndex)
        if tools[r % size] == target:
            return abs(r - startIndex)
        l, r = l-1, r+1
    return -1

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    tools_count = int(input().strip())

    tools = []

    for _ in range(tools_count):
        tools_item = input()
        tools.append(tools_item)

    startIndex = int(input().strip())

    target = input()

    result = toolchanger(tools, startIndex, target)

    fptr.write(str(result) + '\n')

    fptr.close()
