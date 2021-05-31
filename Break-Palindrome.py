#!/bin/python3

import math
import os
import random
import re
import sys


# Complete the 'breakPalindrome' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING palindromeStr as parameter.

def breakPalindrome(palindromeStr):
    size = len(palindromeStr)
    if size > 1:
        returnStr = ""
        for idx, char in enumerate(palindromeStr):
            if char.isalnum() and char > "a" and size // 2 != idx:
                return returnStr + "a" + palindromeStr[idx+1:]
            else:
                returnStr += char
    return "IMPOSSIBLE"      
    

if __name__ == '__main__':