

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
