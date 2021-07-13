#!/bin/python3

import math
import os
import random
import re
import sys


# Complete the maxTickets function below.
from collections import Counter

def maxTickets(tickets):
    t = sorted(tickets)
    longest = 1
    start, end = 0, 1
    while end < len(t):
        if abs(t[end] - t[end-1]) <= 1:
            end += 1 
        else:
            print(longest, start, end)
            diff = end - start 
            longest = max(longest, diff)
            start = end
            end += 1
    return max(longest, end - start)
    
        
    
if __name__ == '__main__':