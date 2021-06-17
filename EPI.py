import math 
import collections
import random
import copy

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



####################################################################################################
#                                           STRINGS                                                #
####################################################################################################


if __name__ == "__main__":

    # ARRAYS

    # count_bits(5)
    # parity(10)
    # extract_lowest_set_bit(20)
    # swap_bits(17, 2, 4)
    # reverse_bits(20)
    # multiply(5, 10)
    # power(2.0, -2)
    # print(is_palindrome(1000021))
    # print(is_rectangle(Point(0, 0), Point(100, 0), Point(0, 200), Point(100, 200)))
    # TwoDcomp()
    # dutch_flag_partition(4, [1, 4, 2, 8, 3, 7, 2, 5, 1, 7])
    print(multiply([1,2,9], [1,1]))

    # STRINGS
