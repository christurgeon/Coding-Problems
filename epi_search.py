import collections 
import math
from typing import List


def binarySearch(t: int, A: List[int]):
    L, R = 0, len(A)-1
    while L <= R:
        M = L + (R - L) / 2
        if A[M] < t:
            L = M + 1
        elif A[M] == t:
            return M
        else:
            R = M - 1
    return -1


def binSearchRecursive(target, array, left, right):
    if left > right:
        return -1
    mid = left + (right - left) // 2   
    if array[mid] < target:
        return binSearchRecursive(target, array, mid + 1, right)
    elif array[mid] == target:
        return mid      
    else: # array[mid] > target 
        return binSearchRecursive(target, array, mid, right - 1)

    
def searchFirstOfK(A: List[int], k: int):
    """
    search for the first instance of k in A, where there can be repeats
    assume A is sorted, use binsearch, as soon as we find k, bring right over to mid - 1
    and update the running result
    """
    left, right, result = 0, len(A)-1, -1
    # A[left:right + 1] is the candidate set
    while left <= right: 
        mid = (left + right) // 2
        if A[mid] > k:
            right = mid - 1
        elif A[mid] == k:
            result = mid 
            right = mid - 1 # nothing to the right of mid can be solution
        else: # A[mid] < k
            left = mid + 1
    return result


def searchEntryEqualToItsIndex(A: List[int]):
    """
    use binsearch, with a left and right index, we adjust left or right
    """
    left, right = 0, len(A)-1
    while left <= right:
        mid = (left + right) // 2
        diff = A[mid] - mid
        # A[mid] == mid if and only if difference == 0
        if diff == 0:
            return mid
        elif diff > 0:
            right = mid - 1
        else: # diff < 0
            left = mid + 1
    return -1


def searchSmallest(A: List[int]):
    """
    find the smallest entry in a cicular sorted array
    use divide and conquer
    """
    left, right = 0, len(A) - 1
    while left < right:
        mid = (left + right) // 2
        if A[mid] > A[right]:
            # minimum must be A[mid + 1:right + 1]
            left = mid + 1
        else: # A[mid] <= A[right]
            # minimum cannot be in A[mid + 1: right + 1] so it must be in A[left:mid + 1]
            right = mid
    return left


def squareRoot(k: int):
    left, right = 0, k 
    # candidates are between [left, right] where everything before left
    # has sqr root <= k and after right is > k
    while left <= right:
        mid = (left + right) // 2
        if mid*mid <= k:
            left = mid + 1
        else:
            right = mid - 1
    return left - 1


def realSquareRoot(x: float):
    """
    take in a floating point number and return its square root,
    same premise as above but not computing with integer division
    """
    # based on x's value relative to 1.0, copmute the search range
    left, right = (x, 1.0) if x < 1.0 else (1.0, x)

    # keeps searching as long as left != right
    while not math.isclose(left, right):
        mid = 0.5 * (left + right)
        if mid*mid > x:
            right = mid 
        else:
            left = mid 
    return left


def sortedMatrixSearch(A: List[List[int]], x: int):
    """
    given number x, check to see if it's within 2D sorted array, 
    column and row are sorted respectively, 
    approach, start from first row rightmost column and if that number is too small then move down a column,
    else move backwards across that row
    """
    row, col = 0, len(A[0]) - 1 # top right corner
    while row < len(A) and col >= 0:
        if A[row][col] == x:
            return True
        elif A[row][col] < x:
            row += 1
        else: # A[row][col] > x 
            col -= 1
    return False


MinMax = collections.namedtuple('MinMax', ('smallest', 'largest'))
def findMinMaxSimultaneously(A: List[int]):
    """
    keep track of min and max in one pass
    """
    def min_max(a, b):
        return MinMax(a, b) if a < b else MinMax(b, a)
    
    if len(A) == 0:
        return None
    if len(A) == 1:
        return MinMax(A[0], A[0])

    global_min_max = min_max(A[0], A[1])
    # process two at a time
    for i in range(2, len(A) - 1, 2):
        local_min_max = min_max(A[i], A[i+1])
        global_min_max = MinMax(min(global_min_max.smallest, local_min_max.smallest), max(global_min_max.largest,  local_min_max.largest))
        # if there is odd number of elements then compare the last
    if len(A) % 2 == 1:
        global_min_max = MinMax(min(global_min_max.smallest, A[-1]), max(global_min_max.largest, A[-1]))
    
    return global_min_max