#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'droppedRequests' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY requestTime as parameter.
#

def droppedRequests(requestTime):
    
    # helper function to avoid code duplication
    def handleDrop(current_request, dropped_tracker, dropped_count, i):
        if current_request not in dropped_tracker or dropped_tracker[current_request] != i:
                dropped_tracker[current_request] = i
                return dropped_count + 1
        return dropped_count 
                
    dropped_count = 0
    dropped_tracker = dict()

    # use the fact of the array being sorted to check
    for i in range(len(requestTime)):
        current_request = requestTime[i]
        if i >= 3 and current_request == requestTime[i-3]:
            dropped_count = handleDrop(current_request, dropped_tracker, dropped_count, i)
        if i >= 20 and current_request - requestTime[i-20] < 10:
            dropped_count = handleDrop(current_request, dropped_tracker, dropped_count, i)
        if i >= 60 and current_request - requestTime[i-60] < 60:
            dropped_count= handleDrop(current_request, dropped_tracker, dropped_count, i)            
            
    return dropped_count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    requestTime_count = int(input().strip())

    requestTime = []

    for _ in range(requestTime_count):
        requestTime_item = int(input().strip())
        requestTime.append(requestTime_item)

    result = droppedRequests(requestTime)

    fptr.write(str(result) + '\n')

    fptr.close()
