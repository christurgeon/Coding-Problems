#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'bestTrio' function below.
#
# The function is expected to return an INTEGER.
# The function accepts UNWEIGHTED_INTEGER_GRAPH friends as parameter.
#

#
# For the unweighted graph, <name>:
#
# 1. The number of nodes is <name>_nodes.
# 2. An edge exists between <name>_from[i] and <name>_to[i].
#
#

def bestTrio(friends_nodes, friends_from, friends_to):
    trios = dict()
    graph = dict()
    for from_, to_ in zip(friends_from, friends_to):
        if from_ not in graph:
            graph[from_] = [to_]
        else:
            graph[from_].append(to_)
        if to_ not in graph:
            graph[to_] = [from_]
        else:
            graph[to_].append(from_)
            
    for f1, f1_friends in graph.items():
        if len(f1_friends) > 2:
            for f2 in f1_friends:
                f2_friends = graph[f2]
                for f3 in f2_friends:
                    f3_friends = graph[f3]
                    trio_key = tuple( sorted([f1, f2, f3]) ) 
                    if trio_key in trios:
                        continue
                    if f3 != f1 and f1 in f3_friends and f2 in f3_friends:
                        trios[trio_key] = len(f1_friends)-2 + len(f2_friends)-2 + len(f3_friends)-2
        
    values = trios.values()
    if len(values) > 0:
        return min(values)
    return -1
                        
            

if __name__ == '__main__':