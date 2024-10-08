import math 
import collections
import random
import itertools
from typing import List, Iterator


class ListNode:
    def __init__(self, data=0, next=None):
        self.data = data
        self.next = next 

class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data 
        self.left = left 
        self.right = right 

        
#############################################################################################################
#############################################################################################################


def count_bits(x):
    n = 0
    while x:
        print(bin(x))
        n += x & 1
        x >>= 1
    return n


def parity(x):
    result = 0
    while x:
        result ^= 1 # XOR to keep track of parity
        x &= x - 1  # drops lowest set bit of x
    return result


def extract_lowest_set_bit(x):
    print(bin(x))
    l = x & ~(x - 1)
    print(bin(l))
    return l


def swap_bits(x, i, j):
    print("Before:", bin(x), x)
    if (x >> i) & 1 != (x >> j) & 1: # if i and j are different, we swap
        mask = (1 << i) | (1 << j)
        print(bin(mask))
        x ^= mask
    print("After:", bin(x), x)
    return x


def lonely_one(i):
    return 1 << i


def reverse_bits(x):
    # brute force, no caching reverses
    result = 0
    while x:
        result <<= 1 # push to the left by one
        result |= x & 1 # add a one if it is there
        x >>= 1 # right shift
        print(bin(result), bin(x))
    return result


def closes_int_same_bit_count(x):
    num_unsigned_bits = 64
    for i in range(num_unsigned_bits - 1):
        if (x >> i) & 1 != (x >> (i + 1)) & 1: # if two consecutives are different
            x ^= (1 << i) | (1 << (i + 1))     # XOR with 1 set to the two indices ki
            return x
    raise ValueError("All bits are 0 or 1")

    
def multiply(a, b):
    def add(a, b):
        print(bin(a), bin(b))
        if b == 0:
            return a 
        else:
            return add(a^b, (a&b) << 1)
    current_sum = 0
    while a:
        print(" a:", bin(a), " b:", bin(b))
        if a & 1:
            current_sum = add(current_sum, b)
        a, b = a >> 1, b << 1
    return current_sum


def power(x: float, y: int):
    result, power = 1.0, y 
    if y < 0: 
        power, x = -power, 1.0 / x
    print("Power:", power, "Bin_power", bin(power), "x:", x)
    while power: 
        if power & 1:
            print("res:", result, result*x)
            result *= x
        x, power  = x*x, power >> 1 
    print(result)
    return result


def reverse(x: int):
    result, remaining = 0, abs(x)
    while remaining:
        result = result * 10 + remaining % 10
        remaining //= 10
    return -result if x < 0 else result


def is_palindrome(x: int) -> bool:
    if x <= 0:
        return x == 0
    else:
        num_digits = math.floor(math.log(x, 10)) + 1
        most_significant = 10 ** (num_digits - 1)
        for i in range(num_digits // 2):
            print("Pre", x, most_significant)
            if x // most_significant != x % 10:
                return False
            x %= most_significant
            x //= 10
            most_significant //= 100 
            print(x, most_significant)
        return True

    
# inclusive of lower, and upper
def uniform_random(lower: int, upper: int) -> int:
    total_outcomes = upper - lower + 1 
    while True:
        result, i = 0, 0
        while (1 << i) < total_outcomes:
            result = (result << 1) | random.random()
            i += 1
        if result < total_outcomes:
            break
    return result + lower

Rect = collections.namedtuple('Rect', ('x', 'y', 'width', 'height'))

"""
x-----x
      | 
 R2   |
    x----x
x-----x  |  
    x----x
"""

def rectangle_intersection(r1: Rect, r2: Rect) -> Rect:
    def does_intersect(r1, r2):
        return r1.x <= r2.x + r2.width  and \
               r2.x <= r1.x + r1.width  and \
               r1.y <= r2.y + r2.height and \
               r2.y <= r1.y + r1.height

    if not does_intersect(r1, r2):
        return Rect(0, 0, -1, -1)
    x_loc = max(r1.x, r2.x)
    y_loc = max(r1.y, r2.y)
    return Rect (
        x_loc, 
        y_loc, 
        min(r1.x+r1.width, r2.x+r2.width) - x_loc,
        min(r1.y+r1.height, r2.y+r2.height) - y_loc
    )

# given four points on a plane, how to determine if there is a rectangle
Point = collections.namedtuple('Point', ('x', 'y'))
def is_rectangle(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    cx = (p1.x + p2.x + p3.x + p4.x) / 4
    cy = (p1.y + p2.y + p3.y + p4.y) / 4
    # no need to sqrt
    d1 = (cx - p1.x)**2 + (cy - p1.y)**2 
    d2 = (cx - p2.x)**2 + (cy - p2.y)**2 
    d3 = (cx - p3.x)**2 + (cy - p3.y)**2 
    d4 = (cx - p4.x)**2 + (cy - p4.y)**2
    return d1 == d2 and d2 == d3 and d3 == d4 


#  list[int]
def even_odd(L) -> None:
    even, odd = 0, len(L)-1 # partition the even to front, odd to back
    while even < odd:
        if L[even] % 2 == 0: # even
            even += 1
        else: # swap 
            L[even], L[odd] = L[odd], L[even] 
            odd -= 1
            # could increment even here too

            
# O(n) to check if element is in an array
"""
list = [1, 2, 3]
element = 2
if element in list:
    print("there")
A = [1, 2, 3]
B = A       # reference
B = list(A) # copy
B = copy.copy(A)
B = copy.deepcopy(A)
A.reverse()     # in-place
reverse(A)      # returns an iterator
A.sort()        # sort in place
sorted(A)       # returns a copy
del A[i]        # deleted the ith element
del[i:j]        # deletes the slice
to rotate a list
A[k:] + A[:k]
"""

def TwoDcomp():
    M = [['A', 'B', 'C'], [1, 2, 3]]
    print( [x for row in M for x in row] )
    return


def dutch_flag_partition(pivot_index, L):
    def swap(i, j):
        L[i], L[j] = L[j], L[i]
    size = len(L)
    pivot_value = L[pivot_index]
    print("Pivot Value:", pivot_value)

    # group smaller elements
    for i in range(size):
        for j in range(i+1, size):
            if L[j] < pivot_value:
                swap(i, j)
                break

    # group larger elements
    for i in reversed(range(size)):
        for j in reversed(range(i)):
            if L[j] > pivot_value:
                swap(i, j)
                break
    print(L)
    # Time O(n^2)
    # Space O(1)

    
# two iterations
def dutch_flag_partition_OofN_time(pivot_index, L):
    def swap(i, j):
        L[i], L[j] = L[j], L[i]
    size = len(L)
    pivot_value = L[pivot_index]
    print("Pivot Value:", pivot_value)
    smaller = 0
    for i in range(size):
        if L[i] < pivot_value:
            swap(i, smaller)
            smaller += 1
    larger = size - 1
    for i in reversed(range(size)):
        if L[i] > pivot_value:
            swap(i, larger)
            larger -= 1
    print(L)
       
        
def dutch_flag_partition_single_pass(pivot_index, L):
    def swap(i, j):
        L[i], L[j] = L[j], L[i]
    # bottom L[:smaller]
    # middle L[smaller:equal]
    # unclassified L[equal:larger]
    # larger L[larger:]
    larger = len(L)
    smaller = 0
    equal = 0
    pivot_value = L[pivot_index]
    while equal < larger:
        if L[equal] < pivot_value:
            swap(equal, smaller)
            smaller += 1
            equal += 1
        elif L[equal] == pivot_value:
            equal += 1
        else: #L[equal] > pivot_value
            larger -= 1
            swap(larger)

            
# [1,2,9] + 1
#   ==> [1,3,0]
# [9,9,9] + 1
# [1, 0, 0, 0]
def plus_one(L):
    L[-1] += 1
    for i in reverse(range(1, len(L))): # -1 -> 1
        if L[i] != 10:
            break
        L[i] = 0
        L[i-1] += 1
    if L[0] == 10:
        L[0] = 1
        L.append(0)
    return L
        

# num1: list[int], num2: list[int]) -> list[int]:
def multiply(num1, num2):
    if (num1[0] < 0) ^ (num2[0] < 0):
        sign = -1
    else:
        sign = 1
    # initialize the array
    result = [0] * (len(num1) + len(num2)) 
    print(result)
    for i in reversed(range(len(num1))):
        for j in reversed(range(len(num2))):
            result[i + j + 1] += num1[i] * num2[j]
            print(result)
            result[i + j] += result[i + j + 1] // 10
            result[i + j + 1] %= 10
            print(result)

    # this is a generator
    # (i for i, x in enumerate(result) if x != 0) 
    result = result[next((i for i, x in enumerate(result) if x != 0), len(result)):] or [0]
    return [sign * result[0]] + result[1:]


def canReachEnd(A):
    """
    idea is to iteratively compute the furthest reach 
    and update that one when we find a further reach
    """
    furthest_reach = 0
    last_index = len(A) - 1
    i = 0
    while i <= furthest_reach and furthest_reach < last_index:
        furthest_reach = max(furthest_reach, A[i] + i)
        i += 1
    return furthest_reach >= last_index
    

def minStepsToLastLocation(A: List[int]) -> int:
    """
    Write a program to compute the minimum amount of steps to reach the last location
    Adapted from EPI
    [1,1,1,1] => 3 
    [1,2,1,1] => 2
    [3,0,0,0] => 1
    """
    size = len(A)
    furthest_we_can_reach = A[0]
    # Track the remaining steps we can make from the current max, if steps runs out, another jump is made...
    steps = A[0]
    jumps = 1
    for i in range(1, size):
        if i == size - 1:
            return jumps
        furthest_we_can_reach = max(furthest_we_can_reach, A[0] + i)
        steps -= 1
        if steps == 0:
            jumps += 1
            if i >= furthest_we_can_reach:
                return -1
            steps = furthest_we_can_reach - 1
    return jumps


def minStepsToLastLocation(A):
    """
    The idea is to iteratively calculate the minimum number of jumps required
    to reach the end of the list. We track the furthest reach in the current 
    jump, and when steps are exhausted, we increment the jump count and update 
    the steps with the difference between the furthest reach and the current index.
    """
    if len(A) == 1:
        return 0
    
    furthest_reach = 0
    last_index = len(A) - 1
    steps = A[0]
    jumps = 0
    i = 0
    
    while i < last_index:
        furthest_reach = max(furthest_reach, A[i] + i)
        steps -= 1
        
        if steps == 0:
            jumps += 1
            if furthest_reach <= i:
                return -1
            steps = furthest_reach - i    
        i += 1
        
        if furthest_reach >= last_index:
            jumps += 1
            break
    
    return jumps
    

def deleteDuplicates(A):
    """
    incrementally write values to the left and skip over others
    """
    if not A:
        return 0

    write_index = 1
    for i in range(1, len(A)):
        if A[write_index - 1] != A[i]:
            A[write_index] = A[i]
            write_index += 1
    return write_index


def buyAndSellStockOnce(P):
    """
    input is prices over time, want to buy and sell one time for max profit, backtest
    iteratively update running min and max profit as we go
    """
    min_price_so_far, max_profit = float('inf'), 0.0
    for price in P:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit


def buyAndSellStockTwice(prices):
    max_total_profit = 0.0
    min_price_thus_far = float('inf')
    first_buy_and_sell_profits = [0.0] * len(prices)
    # for each day, we record maximum profit if we sell on that day

    for i, price in enumerate(prices):
        min_price_thus_far = min(min_price_thus_far, price)
        max_total_profit = max(max_total_profit, price - min_price_thus_far)
        first_buy_and_sell_profits[i] = max_total_profit

    # backward phase, for each day, find the max profit if we make
    # the second buy on that day
    max_price_so_far = float('-inf')
    for i, price in reversed(list(enumerate(prices[1:], 1))):
        max_price_so_far = max(max_price_so_far, price)
        max_total_profit = max(
            max_total_profit,
            max_price_so_far - price + first_buy_and_sell_profits[i]
        )
    
    return max_total_profit


def rearrange(A):
    """
    computing an alteration of max, min, max, in s.t. A[0] >= A[1] <= A[2] >= A[3] ...
    could sort and then choose every other, but easiest is
    rearrange around the median, then perform the interleaving, median finding is O(n)
    """
    for i in range(len(A)):
        A[i:i + 2] = sorted(A[i:i + 2], reverse=bool(i % 2))


def rearrangeV2(A):
    """
    Rearranges the list A such that it follows the pattern:
    A[0] >= A[1] <= A[2] >= A[3] <= A[4] ...

    The function iterates through the list and swaps adjacent elements
    to ensure the pattern is followed.
    """
    n = len(A)
    
    for i in range(1, n):
        if i % 2 == 1:
            # If we're at an odd index (i = 1, 3, 5, ...), ensure A[i-1] >= A[i]
            if A[i - 1] < A[i]:
                A[i - 1], A[i] = A[i], A[i - 1]
        else:
            # If we're at an even index (i = 2, 4, 6, ...), ensure A[i-1] <= A[i]
            if A[i - 1] > A[i]:
                A[i - 1], A[i] = A[i], A[i - 1]

        
def enumeratePrimes(n):
    """
    given n, include all primes up to and including n
    """
    primes = []
    #is_prime[p] represents if p is prime or not, initially all set to true, then use sieving to eliminate
    is_prime = [False, False] + [True] * (n - 1)
    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)
            # sieve p's multiples
            for i in range(p * 2, n + 1, p):
                is_prime[i] = False 
    return primes


def generatePrimes(n):
    """
    improve on previous by seiving p's multiples from p^2 instead of p,
    since all numbers form kp, where k < p have already been seived out
    """
    if n < 2:
        return []
    size = (n - 3) // 2 + 1
    primes = [2] #stores primes from 1 to n
    #is_prime[i] represents (2i + 3) is prime or not
    #for example, is_prime[0] represents 3 is prime or not, is_price[1] represents 5, is_prime[2] represents 7, etc
    #initially set each to true, then use seiving to elikminate nonprimes
    is_prime = [True] * size 
    for i in range(size):
        if is_prime[i]:
            p = i * 2 + 3
            primes.append(p)
        #sieving from p^2, where p^2 = (4i^2 + 12i + 9). The index in is_prime is 
        # (2i^2 + 6i + 3) bc is_prime[i] represents 2i + 3
        #
        # note that we need to use long for j because p^2 might overflow
        for j in range(2 * i**2 + 6*i + 3, size, p):
            is_prime[j] = False 
    return primes


def applyPermutation(perm, A):
    """
    move one element at a time to its correct location, but we need to move the existing elements somewhere
    the natural place to move the existing element is to the recently vacated place, with a swap,
    but then we need to update the permutation array
    # once P[i] == i, we are done with the sequence of swaps
    """
    for i in range(len(A)):
        while perm[i] != i:
            A[perm[i]], A[i] = A[i], A[perm[i]]
            perm[perm[i]], perm[i] = perm[i], perm[perm[i]]


def applyPermutationNoModifyingPermList(perm, A):
    n = len(A)
    for i in range(n):
        # Follow the cycle starting from element i
        next_position = i
        
        while perm[next_position] >= 0:
            # Swap the current element with the target element
            target_position = perm[next_position]

            # Swap the current element with the target element
            A[i], A[target_position] = A[target_position], A[i]
            
            # Mark the position in P as visited by setting it negative
            perm[next_position] -= n
            
            next_position = target_position
    
    # Restore the permutation array
    for i in range(n):
        perm[i] += n


def inversePermutation(perm):
    """
    find the unique inverse of a permutation

    Permutation means that at location i should be element P[i]...
    Inverse permutation means that at element P[i] should be element i

    Example:
        Given P = [2, 0, 1, 3], the inverse permutation P_inv should be [1, 2, 0, 3].
    Explanation:
        P[0] = 2 means that in P_inv, the element 0 should be placed at index 2.
        P[1] = 0 means that in P_inv, the element 1 should be placed at index 0.
        P[2] = 1 means that in P_inv, the element 2 should be placed at index 1.
        P[3] = 3 means that in P_inv, the element 3 should be placed at index 3.
    """
    n = len(perm)
    P_inv = [0] * n
    for i in range(n):
        P_inv[perm[i]] = i    
    return P_inv


def nextPermutation(perm):
    # start from the end and look for largest decresing sequence, then grab the min from the left and replce it, 
    # then reverse the sort on the right seuqence
    inversion = len(perm) - 2
    while inversion >= 0 and perm[inversion] >= perm[inversion + 1]:
        inversion -= 1
    if inversion == -1:
        return []
    for i in reversed(range(inversion + 1, len(perm))):
        if perm[i] > perm[inversion]:
            perm[inversion], perm[i] = perm[i], perm[inversion]
            break 
    # reverse the ordered sequence to get next perm
    perm[inversion + 1:] = reversed(perm[inversion + 1:])
    return perm


def moveAllZerosToBeginningOfArray(A):
    write_index = 0
    for i in range(1, len(A)):
        if A[i] == 0:
            A[write_index], A[i] = A[i], A[write_index]
            write_index += 1
    return A


def moveAllZerosToEndOfArray(A):
    write_index = len(A) - 1
    for i in reversed(range(len(A))):
        if A[i] == 0:
            A[write_index], A[i] = A[i], A[write_index]
            write_index -= 1
    return A


def generateSubsetBySampling(k, A):
    for i in range(k):
        r = random.randint(1, len(A) - 1)
        A[i], A[r] = A[r], A[i]
    # return A[:k + 1] 

    
def reservoirSampling(k, A):
    result = A[:k]
    for i in range(k, len(A)):
        r = random.randrange(i + 1)
        if r < k:
            result[r] = A[i]
    print(result)
    return result


def majorityElementSearch(stream: Iterator[int]) -> str:
    candidate_count = 0
    for it in stream:
        if candidate_count == 0:
            candidate = it
            candidate_count += 1
        elif candidate == it:
            candidate_count += 1
        else:
            candidate_count -= 1
    return candidate


def onlineRandomSampling(stream: Iterator(int), k: int):
    """
    keep track of a sampling out of a data stream
    """
    running_sample = list(itertools.islice(stream, k)) # first k elements
    num_seen = k
    for x in stream:
        num_seen += 1
        idx_for_replace = random.randrange(num_seen)
        if idx_for_replace < k:
            running_sample[idx_for_replace] = x 
    return running_sample


def computePascalTriangle(n: int):
    """
    initialize 2d list then look through the rows and across columns summing
    the two above to create the new value
    """
    result = [[1] * (i + 1) for i in range(n)]
    for i in range(n):
        for j in range(1, i):
            result[i][j] = result[i-1][j-1] + result[i-1][j]
    return result


def isStrAPalindrome(s: str):
    """
    tip: s[~i] is the same as s[-(i+1)]

    the last element at an index is located at s[-1]
    """
    return all(s[i]) == s[~i] for i in range(len(s) // 2))


def ss_decode_col_id(col: str) -> int:
    """
    decode spread sheet columns e.g. AA, ZZZ, BZX to their integer representation

    so A  -> 1
       Z  -> 26
       AA -> 27

    multiply result by 26 because these are essentially base 26 numbers, so we want to shift it over one spot
    """
    result = 0
    for c in col:
        letter_index = ord(c.upper()) - ord('A') + 1
        result = (result * 26) + letter_index
    return result
    

def isBalancedBinaryTree(tree: BinaryTreeNode) -> bool:
    """
    if for each node in the tree, the difference in height of its 
    left and right subtrees is at most 1
    - preorder traversal: O(n)
    - space bounded by height: O(h)
    """
    BalancedStatusWithHeight = collections.namedtuple('BalancedStatusWithHeight', ('balanced', 'height'))

    # first value of the return value indicates if it is balanced
    # second value is the height of the tree
    def checkBalanced(tree):
        if not tree:
            return BalancedStatusWithHeight(balanced=True, height=-1)
        left_result = checkBalanced(tree.left)
        if not left_result.balanced:
            return left_result
        right_result = checkBalanced(tree.right)
        if not right_result.balanced:
            return right_result
        is_balanced = abs(left_result.height - right_result.height) <= 1
        height = max(left_result.height, right_result.height) + 1
        return BalancedStatusWithHeight(is_balanced, height)

    return checkBalanced(tree).balanced


# elements are unique
# rebuild subtree as we identify nodes within it
# TC: O(n)
def RebuildBSTFromPreorder(preorder_sequence: List[int]):
    def helper(lower, upper):
        if root_index[0] == len(preorder_sequence):
            return None 

        root = preorder_sequence[root_index[0]]
        if not lower <= root <= upper:
            return None
        root_index[0] += 1

        # node that the helper updates root_index[0]
        # so the following two calls are critical
        left_subtree = helper(lower, root)
        right_subtree = helper(root, upper)
        return TreeNode(root, left_subtree, right_subtree)
        
    root_index = [0]
    return helper(float('-inf'), float('inf'))

    
"""
    Search for BST violations in a BFS manner, store upper and lower bound
    on the keys stored at the subtree rooted at that node.
"""
def IsBinaryTreeABinarySearchTree(tree):
    queue = collections.deque((tree, float('-inf'), float('inf')))
    while queue:
        node, lower, upper = queue.popleft()
        if node:
            if not (lower <= node.val <= upper):
                return False 
            queue.append( (node.left, lower, node.val) )
            queue.append( (node.right, node.val, upper) )
    return True


"""
    Write a program that takes as input a BST and a value k and returns the first key 
    that would appear in an inorder traversal which is greater than the input value k
        inorder is left, root, right (so we are looking for k's root essentially)

    basically in sorted order take the one to the right of k
"""
def FindFirstNodeGreaterThanK(tree, k):
    current, greater_than_k = tree, None
    while current:
        if current.data > k:
            greater_than_k = current
            current = current.left
        else: # root and all keys in subtree are <= k so skip them
            current = current.right
    return greater_than_k


# traverse down the right subtree and whenever end is reached, we know that is the greatest value
def FindKLargestElementsInABST(tree, k):
    
    def helper(tree):
        if tree and len(result) < k:
            helper(tree.right)
            if len(result) < k:
                result.append(tree.val)
                helper(tree.left)

    result = []
    helper(tree)
    return result       


def countDecreasingSubArrays(A):
    count = 0
    subarray_length = 1
    for i in range(len(A)-1) :
        if (A[i+1] < A[i]):
            subarray_length += 1
 
        # end of subarray, update the result
        else:
            count += (subarray_length * (subarray_length - 1)) // 2
            subarray_length = 1
     
    # clean up case where we end in a subarray 
    if (subarray_length > 1):
        count += (subarray_length * (subarray_length - 1)) // 2
    return count


##############################################################################################
###   LINKED LISTS
##############################################################################################


def searchList(L: ListNode, key: int):
    while L and L.data != key:
        L = L.next
    return L


def insertAfter(node: ListNode, new_node: ListNode):
    new_node.next = node.next
    node.next = new_node

    
def deleteNode(node: ListNode):
    node.next = node.next.next

    
def mergeTwoSortedLists(L1, L2):
    dummy_head = tail = ListNode()
    while L1 and L2:
        if L1.data <= L2.data:
            tail.next = L1 
            L1 = L1.next
        else:
            tail.next = L2 
            L2 = L2.next
        tail = tail.next
    return dummy_head.next


def reverseSublist(L: ListNode, start: int, finish: int):
    dummy_head = sublist_head = ListNode(0, L)
    for _ in range(1, start):
        sublist_head = sublist_head.next
    # reverse sublist
    sublist_iter = sublist_head.next
    for _ in range(finish - start):
        temp = sublist_iter.next
        sublist_iter.next = temp.next
        temp.next = sublist_head.next 
        sublist_head.next = temp

    return dummy_head.next


def listPivoting(l: ListNode, x: int):
    less_head = less_iter = ListNode()
    equal_head = equal_iter = ListNode()
    greater_head = greater_iter = ListNode()
    # populates the three lists
    while l:
        if l.data < x:
            less_iter.next = l
            less_iter = less_iter.next
        elif l.data == x:
            equal_iter.next = l
            equal_iter = equal_iter.next
        else:
            greater_iter.next = l
            greater_iter = greater_iter.next
        l = l.next
    # combine lists 
    greater_iter.next = None 
    equal_iter.next = greater_head.next
    less_iter.next = equal_head.next 
    return less_head.next


def isLinkedListAPalindrome(L: ListNode):
    # find second hald of L
    slow = fast = L
    while fast and fast.next:
        fast, slow = fast.next.next, slow.next
        
    # compares the first hald to the reversedsecond half
    first_iter, second_iter = L, reverse_list(slow)
    while second_iter and first_iter:
        if second_iter.data != first_iter.data:
            return False 
        first_iter = first_iter.next
        second_iter = second_iter.next
    return True
    
    
def evenOddListMerge(L: ListNode):
    if L is None:
        return L

    even_head_dummy, odd_head_dummy = ListNode(), ListNode()
    tails, turn = [even_head_dummy, odd_head_dummy], 0
    while L:
        tails[turn].next = L
        L = L.next
        tails[turn] = tails[turn].next
        turn ^= 1
    tails[1].next = None 
    tails[0].next = odd_head_dummy.next
    return even_head_dummy.next


def hasCycle(head: ListNode):
    """
    use a slow and fast iteration, if fast passes slow
    then a cycle is detected
    """
    fast = slow = head 
    while fast and fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        # cycle detected
        if slow is fast:
            slow = head
            # advance both pointers
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow # start of the cycle
    return None


def overlappingNoCycle(l0: ListNode, l1: ListNode):
    # compute size of both lists, advance longer head to where overlap would start
    # go node by node comparing each one
    
    def length(L):
        length = 0
        while L:
            L = L.next
            length += 1
        return length
    
    l0_len = length(l0)
    l1_len = length(l1)
    if l0_len > l1_len:
        l0, l1 = l1, l0 # arrange so l1 is the longer one
    # advance the longer list to get equal length lists
    for _ in range(abs(l0_len - l1_len)):
        l1 = l1.next
    
    while l0 and l1 and l0 is not l1:
        l0 = l0.next 
        l1 = l1.next
    return l0
    

def overlappingLists(l0: ListNode, l1: ListNode):
    # store the start of the cycle if any
    root0 = hasCycle(l0)
    root1 = hasCycle(l1)
    
    if not root0 and not root1:
        # both lists dont have cycles
        return overlappingNoCycle(l0, l1)
   
    elif (root0 and not root1) or (not root0 and root1):
        # one list has a cycle, no overlapping
        return None
    
    # both lists have cycles
    temp = root1
    while temp:
        temp = temp.next
        if temp is root0 or temp is root1:
            break
            
    return root1 if temp is root0 else None 
    

def deletionFromList(node_to_delete: ListNode):
    node_to_delete.data = node_to_delete.next.data
    node_to_delete.next = node_to_delete.next.next


def removeKthLast(L: ListNode, k: int):
    dummy_head = ListNode(0, L)
    first = dummy_head
    for _ in range(k):
        first = first.next
    second = dummy_head
    while first:
        first, second = first.next, second.next
    # second points to the (k+1)th last node, deltes its successor
    # advance in tandem, when first reaches tail, second is at (k+1)th node and we can remove it
    second.next = second.next.next 
    return dummy_head.next


def removeDuplicates(L: ListNode):
    it = L
    while it:
        # uses next_distinct to find the next distinct value
        next_distinct = it.next
        while next_distinct and next_distinct.data == it.data:
            next_distinct = next_distinct.next
        it.next = next_distinct
        it = next_distinct
    return L


def cyclicallyRightShiftList(L: ListNode, k: int):
    if L is None:
        return L 
    # compute the length of L and the tail
    tail = L
    n = 1
    while tail.next:
        n += 1
        tail = tail.next
    
    k %= n
    if k == 0:
        return L
    
    tail.next = L # makes a cycle by connecting tail to head
    steps_to_new_head = n - k
    new_tail = tail 
    while steps_to_new_head:
        steps_to_new_head -= 1
        new_tail = new_tail.next
    
    new_head = new_tail.next
    new_tail.next = None
    return new_head


##############################################################################################
###   GREEDY & INVARIANTS
##############################################################################################


# gallons[i] is amount of gas in city i
# distances[i] is the distanct i to the next city
# Suppose we pick z as starting point, with gas present at z. Since we never have less gas than we started with at z, and when we return to z we have 0 gas 
#   (since it's given that gas is just enough to complete the traversal)
#  assumes always exists an ample city
def FindAmpleCity(gallons: List[int], distances: List[int]):
    mpg = 20
    remaining_gallons = 0
    city_remaining_gallons_tuple = (0, 0) # (city, gallons)
    cities_count = len(gallons)
    for i in range(1, cities_count):
        # fueled up - fuel used to go to next city
        remaining_gallons += gallons[i-1] - (distances[i-1] // mpg)
        if remaining_gallons < city_remaining_gallons_tuple[1]:
            city_remaining_gallons_tuple = (i, remaining_gallons)
    return city_remaining_gallons_tuple[0] # return city


"""
 Record heights as we go, and then reduce the width by 
 moving the pointer located at the smaller side O(n)
"""
def MostWaterFilled(heights: List[int]):
    l, r, max_area = 0, len(heights) - 1, 0
    while l < r:
        width = r - l
        height = min(heights[l], heights[r])
        max_area = max(max_area, width * height)
        if heights[l] < heights[r]:
            l += 1
        else: 
            r -= 1
    return max_area


# each worker assigned exactly 2 tasks, each one takes fixed amount of time, and tasks are independent
# answer: worker who gets longest outstanding task also gets shortest
def OptimumAssignmentOfTasks(task_durations: List[int]):
    task_durations.sort()
    return [(task_durations[i], task_durations[~i]) for i in range(len(task_durations) // 2)]


def ScheduleToMinimizeWaitTimes(service_times: List[int]) -> int:
    # greedily process in order of shortest times
    service_times.sort()
    total_waiting_time = 0
    for idx, service_time in enumerate(service_times):
        num_remaining = len(service_times) - (idx + 1)
        total_waiting_time += service_time * num_remaining
    return total_waiting_time


def MinimumTimesToVisitAllSchedules(schedules):
    schedules.sort(key=lambda x: x[-1])
    overlapped_until, times = float('-inf'), 0
    for schedule in schedule:
        start, end = schedule[0], schedule[1]
        if start > current:
            overlapped_until = end
            times += 1 
    return times
