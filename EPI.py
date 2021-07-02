import math 
import collections
import random
import copy
from typing import List


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
        
################################################################################################### REVIEW THIS ONE


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

        
def enumeratePrimes(n):
    """
    given n, include all primes up to and including n
    """
    primes = []
    #is_prime[p] represents if p is prime or not, initially all set to trye, then use sieving to eliminate
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
    move one element at a time to its correct location, but we need to move the existing elements somehwere
    the natural place to move the existing element is to the recently vacated place, with a swap,
    but then we need to update the permutation array
    # once P[i] == i, we are done with the sequence of swaps
    """
    for i in range(len(A)):
        while perm[i] != i:
            A[perm[i]], A[i] = A[i], A[perm[i]]
            perm[perm[i]], perm[i] = perm[i], perm[perm[i]]


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


####################################################################################################
#                                        LINKED LIST                                               #
####################################################################################################


class ListNode:
    def __init__(self, data=0, next=None):
        self.data = data
        self.next = next 

        
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
        if slow if fast:
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
    

def deletionFromList():
    pass

def removeKthLast():
    pass 

def cyclicallyRightShiftList(L: ListNode, k: int):
    pass 

def removeDuplicates(L: ListNode):
    pass 
####################################################################################################
#                                           STRINGS                                                #
####################################################################################################





