
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the 'finalInstances' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER instances
#  2. INTEGER_ARRAY averageUtil

def finalInstances(instances, averageUtil):
    index = 0
    system_sleep = False 
    while index < len(averageUtil):
        rate = averageUtil[index]
        print("Index:",index, "Instances:",instances, "Rate:",rate)
        if rate > 60:
            doubled_instances = instances * 2
            if doubled_instances <= 2 * (10**8):
                instances = doubled_instances
                system_sleep = True
        elif rate < 25 and instances > 1:
            instances = math.ceil(instances / 2)
            system_sleep = True
        
        # Adjust index value
        index = index + 11 if system_sleep else index + 1
        system_sleep = False 
    
    return instances

