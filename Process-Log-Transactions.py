#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'processLogs' function below.
#
# The function is expected to return a STRING_ARRAY.
# The function accepts following parameters:
#  1. STRING_ARRAY logs
#  2. INTEGER threshold
#

def recordTransacton(transaction_tracker, user_id):
    if user_id in transaction_tracker:
        transaction_tracker[user_id] += 1
    else:
        transaction_tracker[user_id] = 1   

def processLogs(logs, threshold):
    transaction_tracker = dict()
    for log in logs:
        log = log.split(" ")
        sender, receiver = log[0], log[1]
        if sender == receiver:
            recordTransacton(transaction_tracker, sender)
        else:
            recordTransacton(transaction_tracker, sender)
            recordTransacton(transaction_tracker, receiver)
            
    # optimal solution might be inserting using array bisection algorithm, python has bisect
    # but it doesnt support insertion while specifying a key such as int ...
    
    result = [user_id for user_id, transaction_count in transaction_tracker.items() if transaction_count >= threshold]
    return sorted(result, key=int)
            
if __name__ == '__main__':
