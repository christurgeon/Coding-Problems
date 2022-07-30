""" NOTE
- For a given problem that lends itself to be solved naturally recursively (i.e. N-Queens)
  - if you want optimal solution, try dynamic programming 
  - if you want all solutions, try backtracking

- For quick select algorithm, easiest way to implement is recursively
  - typical way to find order statistic (median, Nth largest, etc.) in linear time (yes, it has a mathematical name) is using quick select
  - time complexity is O(n) on average because on each time we reduce searching range approximately 2 times. 
"""

import re
import heapq
import collections
import bisect
import time
from typing import *
from functools import lru_cache
from functools import cmp_to_key
from random import random

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left 
        self.right = right

class Node:
    def __init__(self, val):
        self.val = val

##############################################################################################
###   [ARRAYS]
##############################################################################################


# https://leetcode.com/problems/contains-duplicate/
def ContainsDuplicate(nums: List[int]) -> bool:
    n = len(nums)
    return n != len(set(nums))


# https://leetcode.com/problems/richest-customer-wealth/
def MaximumWealth(accounts: List[List[int]]) -> int:
    max_wealth = 0
    for i in range(len(accounts)):
        max_wealth = max(max_wealth, sum(accounts[i]))
    return max_wealth


# https://leetcode.com/problems/shuffle-the-array/
def Shuffle(nums: List[int], n: int) -> List[int]:
    result = [None] * 2*n
    for i in range(0, 2*n, 2):
        result[i], result[i+1] = nums[i//2], nums[i//2 + n]
    return result


# https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/
def KidsWithCandies(candies: List[int], extraCandies: int) -> List[bool]:
    maximum = max(candies)
    return [candy + extraCandies >= maximum for candy in candies]
        

# https://leetcode.com/problems/decompress-run-length-encoded-list/
def DecompressRLElist(nums: List[int]) -> List[int]:
    result = []
    for i in range(0, len(nums) - 1, 2):
        result += [nums[i+1]] * nums[i]
    return result
        
        
# https://leetcode.com/problems/arranging-coins/
def ArrangeCoins(n):
    stairs = 1
    while n - stairs >= 0:
        n -= stairs
        stairs += 1
    return stairs - 1      


# https://leetcode.com/problems/squares-of-a-sorted-array/
def SortedSquares(nums: List[int]) -> List[int]:
    result = [0] * len(nums)
    l, r = 0, len(nums) - 1
    while l <= r:
        left, right = abs(nums[l]), abs(nums[r])
        if left > right:
            result[r-l] = left*left
            l += 1
        else:
            result[r-l] = right*right
            r -= 1
    return result


# https://leetcode.com/problems/build-array-from-permutation/
def BuildArray(nums: List[int]) -> List[int]:
        return [nums[nums[i]] for i in range(len(nums))]


# https://leetcode.com/problems/matrix-diagonal-sum/
def DiagonalSum(mat: List[List[int]]) -> int:
    total = 0
    n = len(mat)
    j = n - 1
    for i in range(n):
        total += mat[i][i]
        if i != j:
            total += mat[i][j]
        j -= 1
    return total        
    

# https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/
def SmallerNumbersThanCurrent(nums: List[int]) -> List[int]:
    buckets = [0] * 101
    for num in nums:
        buckets[num] += 1
    prev = 0
    for i, bucket in enumerate(buckets):
        if bucket > 0:
            buckets[i] = prev
            prev += bucket
    return [buckets[num] for num in nums]


# https://leetcode.com/problems/create-target-array-in-the-given-order/   
def CreateTargetArray(nums: List[int], index: List[int]) -> List[int]:
    target = []
    for num, idx in zip(nums, index):
        target.insert(idx, num)
    return target


# https://leetcode.com/problems/running-sum-of-1d-array/
def RunningSum(nums: List[int]) -> List[int]:
    n = len(nums)
    running = [0] * n
    running[0] = nums[0]
    for i in range(1, n):
        running[i] = nums[i] + running[i-1]
    return running


# https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
def TwoSumII(numbers: List[int], target: int) -> List[int]:
    lo, hi = 0, len(numbers) - 1
    while lo < hi:
        total = numbers[lo] + numbers[hi]
        if total == target:
            return [lo+1, hi+1]
        elif total > target:
            hi -= 1
        else:
            lo += 1
    return [-1, -1]


# https://leetcode.com/problems/maximum-average-subarray-i/
def FindMaximumAverageInSlidingWindow(nums: List[int], k: int) -> float:
    total = sum([nums[i] for i in range(k)])
    maximum = total
    for i in range(1, len(nums) - k + 1):
        total = total - nums[i-1] + nums[i+k-1]
        maximum = max(maximum, total)
    return float(maximum) / k


# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def LengthOfLongestSubstring(s: str) -> int:
    if len(s) == 0:
        return 0
    cache = dict()
    size = 0
    for idx, i in enumerate(s):
        if i in cache:
            size = max(size, len(cache))
            starting = cache[i]
            cache = {k: v for k,v in cache.items() if v > starting}
        cache[i] = idx
    return max(size, len(cache))        


# https://leetcode.com/problems/longest-consecutive-sequence/
def LongestConsecutiveInAnArray(nums: List[int]) -> int:
    if not nums:
        return 0
    
    s = set(nums)
    longest = 0
    for num in nums:
        if num in s:
            s.remove(num)
            current_longest = 1
            lo, hi = num-1, num+1
            while lo in s:
                current_longest += 1
                s.remove(lo)
                lo -= 1
            while hi in s:
                current_longest += 1
                s.remove(hi)
                hi += 1
            longest = max(current_longest, longest)
            
    return longest


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
def BestTimeToBuyAndSellStock(prices: List[int]) -> int:
    profit = 0
    running_min = prices[0]
    for price in prices:
        profit = max(profit, price - running_min)
        running_min = min(running_min, price)
    return profit


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
def MaxProfit(prices: List[int]) -> int:
    result = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            result += prices[i] - prices[i-1]
    return result


# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
def SearchRange(nums: List[int], target: int) -> List[int]:
    found = [-1, -1]
    
    # look for left most one
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            found[0] = mid
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
        
    # look for right most one
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            found[1] = mid
            left = mid + 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return found


# https://leetcode.com/problems/single-number/
def SingleNumberV1(nums: List[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result

def SingleNumberV2(nums: List[int]) -> int:
    d = dict()
    for n in nums:
        d[n] = d.get(n, 0) + 1
        if d[n] == 2:
            del d[n]
    return list(d.keys())[0]
    

# https://leetcode.com/problems/plus-one/
def PlusOne(digits: List[int]) -> List[int]:
    digits[-1] = 1 if len(digits) == 0 else digits[-1] + 1
    i = len(digits) - 1
    while i > 0 and digits[i] == 10:
        digits[i] = 0
        digits[i-1] = digits[i-1] + 1
        i -= 1
    if digits[0] == 10:
        digits[0] = 1
        digits.append(0)
    return digits
        

# https://leetcode.com/problems/remove-duplicates-from-sorted-array/
def RemoveDuplicates(nums: List[int]) -> int:
    if len(nums) <= 1:
        return len(nums)
    i, j = 0, 1
    while j < len(nums):
        if nums[i] != nums[j]:
            i += 1
            nums[i] = nums[j]
        j += 1
    return i + 1
        

# https://leetcode.com/problems/maximum-ascending-subarray-sum/
def MaxAscendingSum(nums: List[int]) -> int:
    current_max = float('-inf')
    subarray_max = nums[0]
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            subarray_max += nums[i]
        else:
            current_max = max(current_max, subarray_max)
            subarray_max = nums[i]
    return max(current_max, subarray_max)        


# https://leetcode.com/problems/merge-intervals/
def MergeIntervals(intervals: List[List[int]]) -> List[List[int]]:
    if len(intervals) <= 1:
        return intervals
    intervals = sorted(intervals, key=lambda i: i[0])
    overlapped = [intervals[0]]
    for i in range(1, len(intervals)):
        l, r = intervals[i][0], intervals[i][1]
        if overlapped[-1][1] >= l:
            overlapped[-1][1] = max(overlapped[-1][1], r)
        else:
            overlapped.append([l, r])
    return overlapped
            

# https://leetcode.com/problems/non-decreasing-array/
def CheckPossibility(nums: List[int]) -> bool:
    length = len(nums)
    if length <= 2:
        return True
    changes = False
    for i in range(length-1):
        if nums[i] > nums[i+1]:
            if changes:
                return False
            else:
                changes = True
            if i > 0:
                if nums[i-1] > nums[i+1]: 
                    nums[i+1] = nums[i]
    return True
            

# https://leetcode.com/problems/move-zeroes/
def MoveZeroes(nums: List[int]) -> None:
    chunk_size = i = 0
    for i in range(len(nums)):
        if nums[i] == 0:
            chunk_size += 1
        elif chunk_size >= 1:
            nums[i], nums[i-chunk_size] = nums[i-chunk_size], nums[i]


# https://leetcode.com/problems/merge-sorted-array/
def MergeSortedArraysInPlace(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    write_index = m + n - 1
    m -= 1
    n -= 1
    while n >= 0:
        if m >= 0 and nums1[m] > nums2[n]:
            nums1[write_index] = nums1[m]
            m -= 1
        else:
            nums1[write_index] = nums2[n]
            n -= 1
        write_index -= 1

def MergeTwoSortedArrays(A, B):
    if not A:
        return B
    if not B:
        return A 

    A_size, B_size = len(A), len(B)
    agg = [None] * (A_size + B_size)

    A_index, B_index, agg_index = 0, 0, 0
    while True:
        if A_index == len(A):
            for i in range(B_index, B_size):
                agg[agg_index] = B[i]
                agg_index += 1
            break
        if B_index == len(B):
            for i in range(A_index, A_size):
                agg[agg_index] = A[i]
                agg_index += 1
            break

        if A[A_index] > B[B_index]:
            agg[agg_index] = B[B_index]
            B_index += 1
        elif A[A_index] < B[B_index]:
            agg[agg_index] = A[A_index]
            A_index += 1
        agg_index += 1

    return agg


# if A is non-decreasing order, can use map if out of order
def RemoveDuplicatesFromAnArray(A):
    write_index = 1
    for i in range(1, len(A)):
        if A[write_index - 1] != A[i]:
            A[write_index] = A[i]
            write_index += 1

    return A[:write_index]


# https://leetcode.com/problems/replace-elements-in-an-array/
def ArrayChange(nums: List[int], operations: List[List[int]]) -> List[int]:
    d = {x: i for i, x in enumerate(nums)}
    for x, y in operations:
        nums[d[x]] = y
        d[y] = d[x]
        del d[x]
    return nums

  
def RotateArrayBykSteps(A, k):
    k = k % len(A)
    return A[len(A) - k:] + A[:len(A) - k]


def RotateMatrix(M):
    N = len(M)
    # loop through each of the squares, will give top left x
    for x in range(N // 2):
        # set the bounds of the square
        for y in range(x, N-x-1):
            # top left, bottom left, top right, bottom right | walking upward 
            M[x][y], M[N-y-1][x], M[N-x-1][N-y-1], M[y][N-x-1] = M[y][N-x-1], M[x][y], M[N-y-1][x], M[N-x-1][N-y-1]
  

# https://leetcode.com/problems/rotate-image/
def RotatImageLeetcode(matrix: List[List[int]]) -> None:
    n = len(matrix)
    for i in range(n // 2):
        for j in range(i, n - i - 1):
            #   TOP          RIGHT          BOTTOM          LEFT
            matrix[i][j], matrix[j][~i], matrix[~i][~j], matrix[~j][i] = matrix[~j][i], matrix[i][j], matrix[j][~i], matrix[~i][~j] 


# 1-D RangeSumQuery, prefixing strategy
class NumArray:
    def __init__(self, nums: List[int]):
        self.lookup = [0, nums[0]]
        for i in range(1, len(nums)):
            self.lookup.append(nums[i] + self.lookup[-1])
    def RangeSumQueryImmutable(self, left: int, right: int) -> int:
        return self.lookup[right+1] - self.lookup[left]


# 2-D RangeSumQuery, prefixing strategy
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        COLS = len(matrix[0])
        self.lookup = [[0] * (COLS + 1)]
        for i, row in enumerate(matrix):
            self.lookup.append([0])
            running_row_sum = 0
            for j, value in enumerate(row):
                running_row_sum += value
                self.lookup[i+1].append(self.lookup[i][j+1] + running_row_sum)
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        large_rectangle = self.lookup[row2+1][col2+1]
        upper_rectangle = self.lookup[row1][col2+1]
        left_rectangle = self.lookup[row2+1][col1] 
        intersection = self.lookup[row1][col1] 
        return large_rectangle - upper_rectangle - left_rectangle + intersection


# TLE -> revisit with different data structure 
# https://leetcode.com/problems/range-sum-query-mutable/
class NumArray:
    def __init__(self, nums: List[int]):
        self.sums = [0, nums[0]]
        for i in range(1, len(nums)):
            self.sums.append(nums[i] + self.sums[i])
             
    # O(n) updating cost, not optimal
    def update(self, index: int, val: int) -> None:
        print(self.sums, index, val)
        original_value = self.sums[index+1] - self.sums[index]
        value_to_propagate = val - original_value # could be positive / negative
        if value_to_propagate != 0:
            for j in range(index+1, len(self.sums)):
                self.sums[j] += value_to_propagate

    def sumRange(self, left: int, right: int) -> int:
        return self.sums[right+1] - self.sums[left]


# https://leetcode.com/problems/next-permutation/
def NextPermutation(nums: List[int]) -> None:
    
    def reverse(a, l, r):
        while l < r:
            a[l], a[r] = a[r], a[l]
            l += 1
            r -= 1
        
    # find longest non-increasing suffix
    i = len(nums) - 1
    while i-1 >= 0 and nums[i-1] >= nums[i]:
        i -= 1
    
    # whole thing is non-increasing, so answer is reversed array
    if i == 0:
        return reverse(nums, 0, len(nums)-1)
    
    # indentify the pivot and find the rightmost successor to the pivot
    pivot = i - 1
    successor = 0
    for i in range(len(nums)-1, pivot, -1):
        if nums[i] > nums[pivot]:
            successor = i
            break
            
    # swap with pivot then reverse the suffix
    nums[pivot], nums[successor] = nums[successor], nums[pivot]
    reverse(nums, pivot+1, len(nums)-1)


# https://leetcode.com/problems/search-insert-position/
def SearchInsert(nums: List[int], target: int) -> int:
    l, h = 0, len(nums) - 1
    if target < nums[0]:
        return 0
    if target > nums[-1]:
        return len(nums)
    while l <= h:
        mid = (l + h) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target and nums[mid+1] > target:
            return mid+1
        if nums[mid] > target:
            h = mid - 1
        else:
            l = mid + 1


# https://leetcode.com/problems/fruit-into-baskets/
# what is the length of longest subarray that contains up to two distinct integers
def TotalFruit(fruits: List[int]) -> int:
    count = {}
    unique = result = end = start = 0
    while end < len(fruits):
        # adjust the count and increment the number of unique if we encounter a new one
        count[fruits[end]] = count.get(fruits[end], 0) + 1
        if count[fruits[end]] == 1: 
            unique += 1
        # while we have 3 unique, adjust the start to catch up
        while unique == 3:
            count[fruits[start]] -= 1
            if not count[fruits[start]]: 
                unique -= 1
            start += 1
        result = max(result, end - start + 1)
        end += 1
    return result


# https://leetcode.com/problems/top-k-frequent-elements/
# first get the frequencies then add those frequencies to buckets 
def TopKFrequent(nums: List[int], k: int) -> List[int]:
    counts = Counter(nums)
    buckets = [[] for _ in range (len(nums) + 1)]
    for num in counts:
        count = counts[num]
        buckets[count].append(num)
        
    result = []
    for i in range(len(buckets)-1, -1, -1):
        bucket = buckets[i]
        if bucket:
            result += bucket
        if len(result) >= k:
            return result
    return result


#https://leetcode.com/problems/top-k-frequent-words/
# O(n*log(n))   
def TopKFrequent_BucketSort(words: List[str], k: int) -> List[str]:
    buckets = [[] for _ in range(len(words) + 1)]
    for word, count in collections.Counter(words).items():
        buckets[count].append(word)
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        bucket = buckets[i]
        bucket.sort()
        for word in bucket:
            result.append(word)
            k -= 1
            if k == 0:
                return result
    return result

# O(n*log(k)) by only keeping k elements within the heap
def TopKFrequent_Heap(words: List[str], k: int) -> List[str]:
    frequencies = collections.Counter(words).items() # word: count
    return [word for word, count in heapq.nsmallest(k, frequencies, key = lambda item: (-item[1], item[0]))]


# https://leetcode.com/problems/split-array-into-consecutive-subsequences/
def IsPossible(nums: List[int]) -> bool:
    frequencies = collections.Counter(nums)
    next_nums = collections.defaultdict(int)
    for num in nums:
        # already a part of a subsequence, continue
        if frequencies[num] == 0:
            continue
        
        # we are expecting this number, move expected to the next number
        if next_nums[num] > 0:
            next_nums[num] -= 1
            next_nums[num+1] += 1
            
        # check if next two numbers are available
        elif frequencies[num+1] > 0 and frequencies[num+2] > 0:
            frequencies[num+1] -= 1
            frequencies[num+2] -= 1
            next_nums[num+3] += 1
        
        # we don't have the number we are looking for, and next two numbers are not available
        else:
            return False
        frequencies[num] -= 1
    return True


# https://leetcode.com/problems/jump-game/
def JumpGameI(nums: List[int]) -> bool:
    if len(nums) <= 1:
        return True
    max_can_reach = nums[0]
    current_index = 0
    while max_can_reach < len(nums) - 1 and current_index != max_can_reach:
        max_range = max_can_reach
        for i in range(current_index, max_range + 1):
            max_can_reach = max(max_can_reach, i + nums[i])
        current_index = max_range
    return True if max_can_reach >= len(nums) - 1 else False


# https://leetcode.com/problems/jump-game-ii/
def JumpGameII(nums: List[int]) -> int:
    if len(nums) == 1:
        return 0
    
    n = len(nums)
    queue = deque([(0, 0, 0)]) # (start, end, path_length)

    # first to reach end should have minimum number of jumps using queue
    while queue:
        start, furthest, path_length = queue.popleft()
        new_furthest = furthest
        for i in range(start, min(n, furthest + 1)):
            new_furthest = max(i + nums[i], new_furthest)
        
        # if we have detected that a path can reach the end, return here
        if new_furthest >= n - 1:
            return path_length + 1
        queue.append((furthest, new_furthest, path_length + 1))
            
    return 2**31 - 1


# O(n + log(m))
# https://leetcode.com/problems/search-a-2d-matrix/
def SearchMatrixI(matrix: List[List[int]], target: int) -> bool:
    n, m = len(matrix), len(matrix[0])
    
    def binsearch(row_index, lo, hi):
        while lo <= hi:
            mid = (hi + lo) // 2
            if matrix[row_index][mid] == target:
                return True
            if matrix[row_index][mid] < target:
                return binsearch(row_index, mid+1, hi)
            else:
                return binsearch(row_index, lo, mid-1)
        return False
    
    for i in range(n - 1):
        if matrix[i][0] <= target < matrix[i+1][0]:
            if binsearch(i, 0, m-1):
                return True
    return binsearch(n-1, 0, m-1)


# O(n + m)
# https://leetcode.com/problems/search-a-2d-matrix-ii/
def SearchMatrixII(matrix: List[List[int]], target: int) -> bool:
    n, m = len(matrix), len(matrix[0])
    row, col = n-1, 0
    while row >= 0 and col < m:
        if matrix[row][col] == target:
            return True
        if col < m-1 and matrix[row][col+1] <= target:
            col += 1
        else:
            row -= 1
    return False


# https://leetcode.com/problems/check-if-matrix-is-x-matrix/
def CheckXMatrix(grid: List[List[int]]) -> bool:
    n = len(grid)
    for i in range(n):
        for j in range(n):
            if i == j or i+j == n-1:
                if grid[i][j] == 0:
                    return False
            elif grid[i][j] != 0:
                return False
    return True


# https://leetcode.com/problems/summary-ranges/
def SummaryRanges(nums: List[int]) -> List[str]:
    result = []
    if nums:
        first, second = nums[0], nums[0]
        for i in range(1, len(nums)):
            number = nums[i]
            if number != second + 1:
                result.append(str(first) + ("" if first == second else f"->{second}"))
                first = second = number
            else:
                second = number
        result.append(str(first) + ("" if first == second else f"->{second}"))
    return result


# https://leetcode.com/problems/container-with-most-water/
def MaxArea(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    width = len(height) - 1
    result = 0
    for i in reversed(range(1, width + 1)):
        if height[left] >= height[right]:
            result = max(result, height[right] * i)
            right -= 1
        else:
            result = max(result, height[left] * i)
            left += 1
    return result


# https://leetcode.com/problems/median-of-two-sorted-arrays/
def FindMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    M, N = len(nums1), len(nums2)
    total = (M + N + 1) // 2
        
    # let N be the larger array
    if M > N:
        nums1, nums2 = nums2, nums1
        M, N = N, M
    
    # bin search through smaller array M
    low, high = 0, M
    while low <= high:
        mid_M = low + (high - low) // 2
        mid_N = total - mid_M
        M_left  = float('-inf') if mid_M == 0 else nums1[mid_M-1]
        N_left  = float('-inf') if mid_N == 0 else nums2[mid_N-1]
        M_right = float('inf')  if mid_M == M else nums1[mid_M]
        N_right = float('inf')  if mid_N == N else nums2[mid_N]

        if M_left <= N_right and N_left <= M_right:
            if (M + N) % 2 == 1:
                return max(M_left, N_left)
            else:
                return (max(M_left, N_left) + min(M_right, N_right)) / 2.0

        if M_left > N_right:
            high = mid_M - 1
        else:
            low = mid_M + 1
    
    return -1


# https://leetcode.com/problems/non-overlapping-intervals/
def EraseOverlapIntervals(intervals: List[List[int]]) -> int:
    intervals.sort() # will sort by first, then second if first is tied
    count = 0
    previous_end = intervals[0][1]
    for i in range(1, len(intervals)):
        start, end = intervals[i]
        if start >= previous_end: # non-overlapping
            previous_end = end
        else: # overlapping, update previous end to be the interval we keep (one with lesser end value)
            previous_end = min(previous_end, end)
            count += 1
    return count


##############################################################################################
###   [STRING]
##############################################################################################


# https://leetcode.com/problems/length-of-last-word/
def LengthOfLastWord(s: str) -> int:
    size = 0
    for char in reversed(s):
        if char == " ":
            if size > 0:
                return size 
        else:
            size += 1
    return size


# https://leetcode.com/problems/reverse-string/
def ReverseString(s: List[str]) -> None:
    for i in range(len(s) // 2):
        s[i], s[~i] = s[~i], s[i]


# https://leetcode.com/problems/truncate-sentence/
def TruncateSentence(s: str, k: int) -> str:
    return " ".join(s.split(" ")[:k])


# https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/
def ArrayStringsAreEqual(word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)


# https://leetcode.com/problems/sorting-the-sentence/
def SortSentence(s: str) -> str:
    d = [None] * 10
    for word in s.split(" "):
        index = int(word[-1])
        d[index] = word[:-1]
    return " ".join([d[i] for i in range(10) if d[i]])
        
        
# https://leetcode.com/problems/split-a-string-in-balanced-strings/
def BalancedStringSplit(s: str) -> int:
    l, r, total = 0, 0, 0
    for char in s:
        if char == 'L':
            l += 1
        else: # char == 'R'
            r += 1
            
        if l == r:
            total += 1
    return total
        
        
# https://leetcode.com/problems/final-value-of-variable-after-performing-operations/
def FinalValueAfterOperations(operations: List[str]) -> int:
    x = 0
    for operation in operations:
        if operation[0] == "+" or operation[-1] == "+":
            x += 1                
        elif operation[0] == "-" or operation[-1] == "-":
            x -= 1
    return x


# https://leetcode.com/problems/maximum-number-of-words-found-in-sentences/
def MostWordsFound(sentences: List[str]) -> int:
    return max(sentence.count(" ") for sentence in sentences) + 1


# https://leetcode.com/problems/remove-palindromic-subsequences/
"""
Given a string consisting of only a's and b's what is the minimum 
amount of palindromes you can remove to make it empty
    solution: since its two chars if s is not palindrome, then it takes 2 removals only
"""
def RemovePalindromicString(s: str) -> int:
    if s == s[::-1]:
        return 1
    return 2


# https://leetcode.com/problems/remove-vowels-from-a-string/
def RemoveVowelsFromAString(S):
    vowels = dict.fromkeys(["a", "e", "i", "o", "u"])
    new_string = []
    for c in S:
        if c not in vowels:
            new_string.append(c)
    return ''.join(new_string)


# https://leetcode.com/problems/defanging-an-ip-address/
def DefangingAnIPAddress(IP):
    split_IP = IP.split('.')
    return "[.]".join(split_IP)


# https://leetcode.com/problems/jewels-and-stones/
# find how many of the stones you have are also jewels
def JewelsAndStones(jewels, stones):
    d = dict.fromkeys(list(jewels))
    cnt = 0
    for stone in stones:
        if stone in d:
            cnt += 1
    return cnt


def IntegerToString(x: int):
    negative = False
    if x < 0:
        x, negative = -x, True
    s = []
    while True:
        s.append(chr(ord('0') + x % 10))
        x = x // 10
        if x == 0:
            break
    # add negative sign back if needed
    return ('-' if negative else '') + ''.join(reversed(s))


# https://leetcode.com/problems/string-to-integer-atoi/
def StringToInteger(S):
    if not S or len(S) == 0:
        return None 
    sign = 1 if S[0] == "+" or S[0].isdigit() else -1
    num = 0
    place = 1
    for c in reversed(S):
        if c.isdigit():
            num += int(c) * place 
            place *= 10
    return sign * num


# https://leetcode.com/problems/implement-strstr/
def StrStr(haystack: str, needle: str) -> int:
    if needle == "":
        return 0
    i = 0
    while i < len(haystack) - len(needle) + 1:
        if haystack[i] == needle[0]:
            if haystack[i:i + len(needle)] == needle:
                return i
        i += 1
    return -1


# https://leetcode.com/problems/multiply-strings/
def Multiply(num1: str, num2: str) -> str:
    result = [0] * (len(num1) + len(num2))
    for i in reversed(range(len(num1))):
        for j in reversed(range(len(num2))):
            result[i + j + 1] += int(num1[i]) * int(num2[j])
            result[i + j] += result[i + j + 1] // 10
            result[i + j + 1] %= 10
    result = ''.join([str(i) for i in result])
    result = result.lstrip("0")
    if result == "":
        result = "0"
    return result
        

# https://leetcode.com/problems/longest-common-prefix/
def LongestCommonPrefix(strs: List[str]) -> str:
    prefix = [""]
    for index, char in enumerate(strs[0]):
        for j in range(1, len(strs)):
            if index == len(strs[j]) or char != strs[j][index]:
                return "".join(prefix)
        prefix.append(char)
    return "".join(prefix)


# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def LongestSubstringWithoutRepeatingCharacters(S):
    if len(S) == 0:
        return 0
    size = 0
    cache = dict()
    for idx, c in enumerate(S):
        if c in cache:
            size = max(size, len(cache))
            starting_index = cache[c]
            cache = {k: v for k, v in cache.items() if v > starting_index}
        cache[c] = idx
    return max(size, len(cache))


# https://leetcode.com/problems/first-unique-character-in-a-string/
def FirstUniqueChar(s: str) -> int:
    d = collections.Counter(s)
    for idx, char in enumerate(s):
        if d[char] == 1:
            return idx
    return -1


# https://leetcode.com/problems/valid-palindrome/
def IsPalindrome(s: str) -> bool:
    l, r = 0, len(s) - 1
    while l <= r:
        left, right = s[l].lower(), s[r].lower()
        if not left.isalnum():
            l += 1
        elif not right.isalnum():
            r -= 1
        elif left == right:
            l += 1
            r -= 1
        else:
            return False
    return True
               

# https://leetcode.com/problems/longest-palindromic-substring/
def LongestPalindromicSubstring(s):
    n = len(s)
    l = r = longest = -1
    for _min in range(n):
        x = 0
        while _min-x >= 0 and _min+x < n:
            if s[_min-x] != s[_min+x]:
                break
            length = 1 + 2 * x
            if length > longest:
                longest = length
                l, r = _min-x, _min+x
            x += 1

    for _min in range(n - 1):
        x = 1
        while _min-x+1 >= 0 and _min+x < n:
            if s[_min-x+1] != s[_min+x]:
                break
            length = 2 * x
            if length > longest:
                longest = length
                l, r = _min-x+1, _min+x
            x += 1

    return s[l:r+1]


# https://leetcode.com/problems/minimum-window-substring/
def MinimumWindowSubstring(s, target):
    counts = collections.Counter(target)
    start, found = 0, 0
    minL, minR = float('-inf'), float('inf')
    for end in range(len(s)):
        
        # keep trying to fill the range
        if s[end] in counts:
            if counts[s[end]] > 0:
                found += 1
            counts[s[end]] -= 1
        
        # we have a substring, record its size then keep trying to increment start
        while found == len(target):
            size = end - start + 1
            if size < minR - minL:
                minL, minR = start, end + 1
                
            # if we see a target character, increment it, if it's now above 0 then break out of loop
            if s[start] in counts:
                counts[s[start]] += 1
                if counts[s[start]] > 0:
                    found -= 1
            start += 1
            
    return s[minL : minR] if minL > float('-inf') else ""


# https://leetcode.com/problems/increasing-decreasing-string/
def SortString(s: str) -> str:
    freqs = sorted([char, count] for char, count in collections.Counter(s).items())
    result = []
    while len(result) != len(s):
        for i in range(len(freqs)):
            if freqs[i][1]:
                freqs[i][1] -= 1
                result.append(freqs[i][0])
        for i in reversed(range(len(freqs))):
            if freqs[i][1]:
                freqs[i][1] -= 1
                result.append(freqs[i][0])
    return ''.join(result)


# https://leetcode.com/problems/greatest-english-letter-in-upper-and-lower-case/
def GreatestLetter(s: str) -> str:
    lookup, answer = set(), ""
    for c in s:
        if c in lookup:
            answer = c.upper() if not answer or c.upper() > answer else answer
        elif c == c.lower():
            lookup.add(c.upper())
        else:
            lookup.add(c.lower())
    return answer


# https://leetcode.com/problems/largest-number/
def LargestNumberMadeFromNums(nums: List[int]) -> str:

    # sorted by value of concatenated string increasingly
    def cmp(x, y):
        if x + y > y + x:
            return 1
        return -1
        
    nums = [str(num) for num in nums]
    nums.sort(key=cmp_to_key(cmp), reverse=True)
    return str(int(''.join(nums)))


# https://leetcode.com/problems/valid-anagram/
def IsAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    leftover = {}
    for c1, c2 in zip(s, t):
        leftover[c1] = leftover.get(c1, 0) + 1
        leftover[c2] = leftover.get(c2, 0) - 1
    return not any(value != 0 for value in leftover.values())


##############################################################################################
###   [LINKED LISTS]
##############################################################################################


# https://leetcode.com/problems/remove-duplicates-from-sorted-list/
def DeleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    current = head
    while current:
        while current.next and current.val == current.next.val:
            current.next = current.next.next
        current = current.next
    return head


# https://leetcode.com/problems/remove-linked-list-elements/
def RemoveElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    dummy_head = ListNode(-1) 
    dummy_head.next = head
    current_node = dummy_head
    while current_node.next:
        if val == current_node.next.val:
            current_node.next = current_node.next.next
        else:
            current_node = current_node.next
    return dummy_head.next
        

# https://leetcode.com/problems/remove-nth-node-from-end-of-list/
def RemoveNthFromEnd(head: ListNode, n: int) -> ListNode:
    size = 0
    pointer = head
    while pointer:
        pointer = pointer.next
        size += 1    
    idx = size - n + 1
    if n == size:
        head = head.next
        return head
    pointer = head
    for _ in range(idx - 2):
        pointer = pointer.next
    node_to_remove = pointer.next
    pointer.next = node_to_remove.next
    return head


# https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/
def GetDecimalValue(head: ListNode) -> int:
    value = []
    while head:
        value.append(str(head.val))
        head = head.next
    return int(''.join(value), 2)


# https://leetcode.com/problems/merge-nodes-in-between-zeros/
def MergeNodes(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy_head = ListNode(0, head)
    writer = dummy_head.next
    current = dummy_head.next.next
    
    while current:
        if current.val == 0:
            if current.next:
                writer = writer.next
                writer.val = 0
        else:
            writer.val += current.val
        current = current.next
        
    writer.next = None
    return dummy_head.next


# https://leetcode.com/problems/rotate-list/
def RotateRight(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head or not head.next or k == 0:
        return head
    
    size = 1
    temp = head
    while temp.next:
        temp = temp.next
        size += 1
    if k % size == 0:
        return head
        
    first, second = head, head
    for _ in range(k % size):
        second = second.next
    while second.next:
        first, second = first.next, second.next
    temp = first.next    
    first.next = None
    second.next = head
    return temp


# https://leetcode.com/problems/reverse-linked-list/
def ReverseLinkedList(head: Optional[ListNode]) -> Optional[ListNode]:
    previous = None
    current = head
    while current:
        temp = current.next
        current.next = previous
        previous, current = current, temp
    return previous

def ReveriseLinkedLstRecursive(head: Optional[ListNode]) -> Optional[ListNode]:
    def helper(previous, current):
        if not current:
            return previous
        temp = current.next
        current.next = previous
        return helper(current, temp)
    return helper(None, head)


# take the spot of the next node
def DeleteNodeFromList(node: ListNode):
    if not node.next:
        node = None 
    else: 
        node.val = node.next.val 
        node.next = node.next.next


# https://leetcode.com/problems/merge-two-sorted-lists/
def MergeTwoLinkedLists(l1: ListNode, l2: ListNode):
    dummy_head = it = ListNode()
    while l1 and l2:
        if l1.val < l2.val:
            it.next = l1 
            l1 = l1.next 
        else:
            it.next = l2
            l2 = l2.next
    if l1:
        it.next = l1
    if l2:
        it.next = l2
    return dummy_head.next


def LinkedListCycle(head: ListNode):
    # detected a cycle if faster iter passes slower iter
    slow = fast = head
    while fast and fast.next and fast.next.next: # fast is if head is None type
        if slow is fast:
            return True
        slow = slow.next
        fast = fast.next.next
    return False


# https://leetcode.com/problems/intersection-of-two-linked-lists/
# Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.
def GetIntersectionNode(headA: ListNode, headB: ListNode):
    """
        - Idea 1: hash table O(n) space 
        - Idea 2: O(n) search O(1) space, align both at same start
    """
    def length(L):
        count = 0
        dummy = L
        while dummy:
            dummy = dummy.next 
            count += 1
        return count

    size_a, size_b = length(headA), length(headB)
    longer = headA if size_a >= size_b else headB 
    shorter = headA if size_a < size_b else headB
    for _ in range(abs(size_a - size_b)):
        longer = longer.next

    while longer and shorter:
        if longer == shorter:
            return longer
        longer, shorter = longer.next, shorter.next 

    return None

# when each one reaches the end, swap list that way they both end up aligned and
# if there is no intersection than at the end they will both be equal to None
def GetIntersectionNodeOptimal(headA: ListNode, headB: ListNode):
    a_iter, b_iter = headA, headB
    while a_iter is not b_iter:
        a_iter = a_iter.next if a_iter else headB
        b_iter = b_iter.next if b_iter else headA
    return a_iter


# https://leetcode.com/problems/reverse-linked-list-ii/
def ReverseBetweenRange(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    if left == right:
        return head
    
    # set up start one behind the left location
    dummy = ListNode(0, head)
    start = dummy
    for _ in range(left-1):
        start = start.next
    
    # reverse the nodes within the range
    prev = None
    curr = start.next
    for _ in range(right-left+1):
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
        
    # take the last node in the range and point it to first node outside of range
    start.next.next = curr
    # point the 'start' to the new first node in the range
    start.next = prev
    return dummy.next


# https://leetcode.com/problems/merge-k-sorted-lists/
def MergeKSortedLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    q = []
    dummy_head = ListNode(None)
    current = dummy_head 
    for i, node in enumerate(lists):
        if node:
            q.append((node.val, i, node))
    heapq.heapify(q)
        
    while q:
        _, i, node = heapq.heappop(q)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(q, (node.next.val, i, node.next))
    
    return dummy_head.next


"""
Another option for the next two problems is to do two traversals,
one time to get the length and a second time to get the middle
"""
# https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/
def DeleteMiddleNode(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy_head = ListNode(0, head)
    slow, fast = dummy_head, dummy_head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    slow.next = slow.next.next
    return dummy_head.next


# https://leetcode.com/problems/middle-of-the-linked-list/
def MiddleNode(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy_head = ListNode(0, head)
    slow, fast = dummy_head, dummy_head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow.next


# https://leetcode.com/problems/swap-nodes-in-pairs/
def SwapPairs(head: ListNode) -> ListNode:
    if head is None or head.next is None:
        return head
    
    dummy_head = head.next
    prior = None
    first = head
    second = head.next
    while first and second:
        first_cache = first
        second_cache = second

        temp = second.next
        second.next = first
        first.next = temp
        first = temp
        
        if temp is not None:
            second = temp.next
        else:
            second = None
            
        if prior:
            prior.next = second_cache
        prior = first_cache
        
    return dummy_head


# https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/
def PairSumInList(head: Optional[ListNode]) -> int:
    
    def reverseList(node: ListNode):
        prev = None
        curr = node
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
    
    # get to the middle node, then reverse the rest
    dummy_head = ListNode(0, head)
    slow, fast = dummy_head, dummy_head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    middle_node = slow.next
    slow.next = reverseList(middle_node)
    
    # loop through start and reversed second half, keeping track of total
    total = 0
    fst, scd = head, slow.next
    while scd:
        total = max(total, fst.val + scd.val)
        fst, scd = fst.next, scd.next
    return total


##############################################################################################
###   [STACKS AND QUEUES]
##############################################################################################


def PolishNotation(S):
    results = []
    delimiter = ',' 
    operations = {
        '+': lambda x, y: x + y, '-': lambda x, y: x - y,
        '*': lambda x, y: x * y, '/': lambda x, y: x / y
    }

    for token in S.split(delimiter):
        if token in operations:
            results.append( operations[token](results.pop(), results.pop()) )
        else:
            results.append(int(token))
    return results[-1]


# https://leetcode.com/problems/evaluate-reverse-polish-notation/
def EvalRPN(tokens: List[str]) -> int:
    results = []
    operations = {
        '+': lambda x, y: x + y, 
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y, 
        '/': lambda x, y: int(float(x) / y)
    }

    for token in tokens:
        if token in operations:
            v1, v2 = results.pop(), results.pop()
            results.append( operations[token](v2, v1) )
        else:
            results.append(int(token))
    return results[-1]
        

# https://leetcode.com/problems/valid-parentheses/
def ValidParenthesis(S):
    lookup = { '{': '}', '[': ']', '(': ')' }
    stack = []
    for c in S:
        if c in lookup:
            stack.append(lookup[c])
        elif len(stack) == 0 or c != stack.pop():
            return False 
    return len(stack) == 0


# https://leetcode.com/problems/min-stack/
class MinStack:
    def __init__(self):
        # tuple (value, curr_min)
        self.__stack = []
    def push(self, val):
        minimum = self.getMin()
        self.__stack.append( (val, min(val, minimum)) )
    def pop(self):
        return self.__stack.pop()
    def top(self):
        if (len(self.__stack) > 0):
            return self.__stack[-1][0]
        return None
    def getMin(self):
        if len(self.__stack):
            return self.__stack[-1][1]
        return 2**31 - 1


# https://leetcode.com/problems/implement-stack-using-queues/
from collections import deque
class MyStack:
    def __init__(self):
        self.q = deque()
    def push(self, x: int) -> None:
        # add to end then reverse order
        self.q.append(x)
        q_len = len(self.q)
        while q_len > 1:
            self.q.append(self.q.popleft())
            q_len -= 1
    def pop(self) -> int:
        return self.q.popleft()
    def top(self) -> int:
        return self.q[0]
    def empty(self) -> bool:
        return len(self.q) == 0


# https://leetcode.com/problems/simplify-path/
def AbsoluteToCanonicalPath(path):
    tokens = [token for token in path.split("/") if token not in [".", ""]]
    stack = []
    for path_part in tokens:
        if path_part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(path_part)
    return "/" + "/".join(stack)


# https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
def RemoveDuplicatesI(s: str) -> str:
    stack = []
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    return ''.join(stack)


# https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
# O(n*k)
def RemoveDuplicatesII(s: str, k: int) -> str:
    stack = []
    last_seen, last_seen_count = '?', -1
    
    i = 0
    while i < len(s):
        char = s[i]
        if char == last_seen:
            # delete k-1 occurences of our character, and do not add it to the stack
            if last_seen_count == k-1:
                for _ in range(k-1):
                    stack.pop()
                # update the last_seen and last_seen_count
                last_seen, last_seen_count = '?', -1
                if stack:
                    last_seen, last_seen_count = stack[-1], 1
                    for j in range(1, min(k, len(stack))):
                        if stack[~j] != last_seen:
                            break
                        last_seen_count += 1
            else:
                last_seen_count += 1
                stack.append(char)
        else:
            last_seen, last_seen_count = char, 1
            stack.append(char)
        i += 1
            
    return ''.join(stack)

# O(n)
def RemoveDuplicates_Optimized(s: str, k: int) -> str:
    stack = [['?', -1]]
    
    # keep track of char and number of consecutive occurences in array
    for char in s:
        if char == stack[-1][0]:
            if stack[-1][1] == k-1:
                stack.pop()
            else:
                stack[-1][1] += 1
        else:
            stack.append([char, 1])
            
    # build the result, make sure we get frequencies right
    result = []
    for i in range(1, len(stack)):
        result += [stack[i][0]] * stack[i][1]
    return ''.join(result)


# https://leetcode.com/problems/132-pattern/
def Find132pattern(nums: List[int]) -> bool:
    stack = [] # (value, minimum seen before it), stack is in monotonically decreasing order
    current_minimum = nums[0]
    n = len(nums)
    for i in range(1, n):
        value = nums[i]
        while stack and value >= stack[-1][0]:
            stack.pop()
        if stack and value > stack[-1][1]:
            return True
        stack.append([value, current_minimum])
        current_minimum = min(current_minimum, value)
    return False


##############################################################################################
###   [TREES]
##############################################################################################

#        4
#     2     6
#    1 3   5  7
#

# in-order: left, root, right
# pre-order: root, left, right
# post-order: left, right, root

def BinaryTreeInOrderTraversal(root):
    if root:
        return BinaryTreeInOrderTraversal(root.left) + [root.val] + BinaryTreeInOrderTraversal(root.right)
    else:
        return []

def BinaryTreePreorderTraversal(root):
    if root:
        return  [root.val] + BinaryTreePreorderTraversal(root.left) + BinaryTreePreorderTraversal(root.right)
    else:
        return []

def BinaryTreePostOrderTraversal(root):
    if root:
        return BinaryTreePostOrderTraversal(root.left) + BinaryTreePostOrderTraversal(root.right) + [root.val]
    else:
        return []


def SearchBinarySearchTree(node, search_value):
    if not node or node.val == search_value:
        return node
    elif node.val > search_value:
        return SearchBinarySearchTree(node.left, search_value)
    else: # node.val < search_value
        return SearchBinarySearchTree(node.right, search_value)


# https://leetcode.com/problems/same-tree/
def SameTree(lhead: TreeNode, rhead: TreeNode):
    if not lhead and not rhead:
        return True
    if (not lhead) ^ (not rhead):
        return False 
    if lhead.val != rhead.val:
        return False
    return SameTree(lhead.left, rhead.left) and SameTree(lhead.right, rhead.right) 


# https://leetcode.com/problems/range-sum-of-bst/
def RangeSumBST(root: Optional[TreeNode], low: int, high: int) -> int:
    total = 0
    def traverse(node):
        nonlocal total
        if node:
            traverse(node.left)
            if low <= node.val <= high:
                total += node.val
            # only bother looking down right subtree if potential values in range
            if node.val <= high:
                traverse(node.right)
    traverse(root)
    return total


# https://leetcode.com/problems/kth-smallest-element-in-a-bst/
def KthSmallest(root: Optional[TreeNode], k: int) -> int:
    result, found = float('-inf'), False
    
    def dfs(node):
        nonlocal result, found, k
        if not node :
            return 2**31 - 1
        if found:
            return result
        
        dfs(node.left)
        k -= 1
        if k == 0:
            found = True
            result = node.val
        dfs(node.right)
        return result
        
    return dfs(root)


# https://leetcode.com/problems/cousins-in-binary-tree/
def IsCousins(root: TreeNode, x: int, y: int) -> bool:

    def helper(root, parent_value, depth):
        if not root:
            return []
        elif root.val == x or root.val == y:
            return [(parent_value, depth)]
        else:
            return helper(root.left, root.val, depth + 1) + helper(root.right, root.val, depth + 1)

    if root:
        vals = helper(root, root.val, 0)
        if len(vals) == 2:
            return vals[0][0] != vals[1][0] and vals[0][1] == vals[1][1]
    return False
        

# https://leetcode.com/problems/binary-tree-paths/
def BinaryTreePaths(root: Optional[TreeNode]) -> List[str]:
    stack = [(root, "")]
    result = []
    while stack:
        node, path = stack.pop()
        if node:
            s = path + str(node.val)
            if not node.left and not node.right:
                result.append(s)
            else:
                s += "->"
                stack.append((node.right, s))
                stack.append((node.left, s))    
    return result


# https://leetcode.com/problems/invert-binary-tree/
def InvertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if root:
        root.left, root.right = root.right, root.left
        _ = InvertTree(root.left)
        _ = InvertTree(root.right)
    return root


# https://leetcode.com/problems/evaluate-boolean-binary-tree/
def EvaluateTree(root: Optional[TreeNode]) -> bool:
    if not root.left and not root.right:
        return root.val
    if root.val == 2:
        return self.evaluateTree(root.left) or self.evaluateTree(root.right)
    else:
        return self.evaluateTree(root.left) and self.evaluateTree(root.right)
            
            
# https://leetcode.com/problems/find-a-corresponding-node-of-a-binary-tree-in-a-clone-of-that-tree/
def GetTargetCopy(original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    node = None
    
    def traverse(tree):
        nonlocal node
        if not tree:
            return
        if tree and tree.val == target.val:
            node = tree
            return 
        traverse(tree.left)
        traverse(tree.right)
    
    traverse(cloned)
    return node


# https://leetcode.com/problems/subtree-of-another-tree/
def IsSubtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    
    def same(tree, subtree):
        if tree and subtree and tree.val == subtree.val:
            return same(tree.left, subtree.left) and same(tree.right, subtree.right)
        elif not subtree and not tree:
            return True
        else:
            return False

    if not root:
        return False 
    if not subRoot:
        return True
    if same(root, subRoot):
        return True
    return IsSubtree(root.left, subRoot) or IsSubtree(root.right, subRoot)


# https://leetcode.com/problems/recover-binary-search-tree/
def RecoverTree(root: Optional[TreeNode]) -> None:

    def inorder(node):  
        nonlocal first, second, previous
        if not node:
            return

        inorder(node.left)  
        if previous and node.val < previous.val:
            if not first:
                first = previous
            # set second to the current node, they need to be swapped, but then 
            # as we iterate if we discover a better candidate, set it there                                     
            if first:
                second = node                    
        previous = node
        inorder(node.right)
        
    first, second, previous = None, None, None   
    inorder(root)        
    if first and second:
        first.val, second.val = second.val, first.val


# https://leetcode.com/problems/symmetric-tree/
def SymmetricTree(root):
    def isSym(left, right):
        if not left and not right:
            return True
        elif left and right:
            return left.val == right.val and isSym(left.left, right.right) and isSym(left.right, right.left)
        else:
            return False
    return not root or isSym(root.left, root.right) 
    

# https://leetcode.com/problems/maximum-depth-of-binary-tree/
def MaximumDepthOfBinaryTree(root):
    def traverse(node, depth):
        if node:
            return max(traverse(node.left, depth + 1), traverse(node.right, depth + 1))
        return depth
    return traverse(root, 0)

# iterative example from leetcode discussion board, for reference 
def MaximumDepthOfBinaryTree_Iteative(root):
    depth = 0
    level = [root] if root else []
    while level:
        depth += 1
        queue = []
        for el in level:
            if el.left:
                queue.append(el.left)
            if el.right:
                queue.append(el.right)
        level = queue
    return depth


# https://leetcode.com/problems/validate-binary-search-tree/solution/
def ValidateBinarySearchTree(root, less=float('inf'), greater=float('-inf')):
    if not root:
        return True 
    if root.val <= greater or root.val >= less:
        return False
    return ValidateBinarySearchTree(root.left, min(less, root.val), greater) and \
           ValidateBinarySearchTree(root.right, less, max(greater, root.val))


# https://leetcode.com/problems/binary-tree-level-order-traversal/
def LevelOrderTreeTraversal(root):
    result, q = [], [root]
    if not root:
        return result
    while q:
        level = []
        result.append([node.val for node in q])
        for node in q:
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        q = level
    return result


# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
def ZigZagLevelOrderTravesal(root):
        result, q = [], [root]
        if not root:
            return []
        left_to_right = True
        while q:
            level = []
            if left_to_right:
                result.append([node.val for node in q])
            else:
                result.append([node.val for node in reversed(q)])
            for node in q:
                if node.left:
                    level.append(node.left)
                if node.right:
                    level.append(node.right)
            q = level 
            left_to_right = not left_to_right
        return result


# https://leetcode.com/problems/path-sum/
def HasPathSum(root: Optional[TreeNode], targetSum: int) -> bool:
    if not root:
        return False

    def helper(node, current_sum):
        if node and not node.left and not node.right and current_sum + node.val == targetSum:
            return True
        elif node and (node.left or node.right):
            return helper(node.left, current_sum + node.val) or helper(node.right, current_sum + node.val)
        return False

    return helper(root, 0) 
        

# https://leetcode.com/problems/binary-tree-maximum-path-sum/
class Solution:
    def __init__(self):
        self.max_path_sum = float('-inf')

    def MaxPathSum(self, root: Optional[TreeNode]) -> int:
        # i think we have to traverse all O(n) time, let's try recursively
        # do bottom up search and trim paths
        def search(node):
            if not node:
                return 0

            # search subtrees and update max path sum
            l, r = search(node.left), search(node.right)
            potential_max_path_sum = l + node.val + r
            self.max_path_sum = max(self.max_path_sum, potential_max_path_sum)
            
            # greatest path from bottom up to current node
            bottom_up_max_sum = node.val + max(l, r)
            return max(bottom_up_max_sum, 0)
        
        search(root)
        return self.max_path_sum    


# https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
def SortedArrayToBST(nums: List[int]) -> Optional[TreeNode]:
    
    def build(nums, lower, upper):
        if lower == upper:
            return None
        middle = (lower + upper) // 2
        node = TreeNode(nums[middle])
        node.left = build(nums, lower, middle)
        node.right = build(nums, middle+1, upper)
        return node

    return build(nums, 0, len(nums))
                      

# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
def LowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

    # store ancestor of each node
    stack = [root]
    parent = {root: None}
    while p not in parent or q not in parent:
        current = stack.pop()
        if current.left:
            parent[current.left] = current
            stack.append(current.left)
        if current.right:
            parent[current.right] = current
            stack.append(current.right)
    
    # traverse upwards from parent to see first parent they have in common
    ancestors = set()
    while p:
        ancestors.add(p)    
        p = parent[p]
    while q not in ancestors:
        q = parent[q]
    return q

def LowestCommonAncestorBST(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # as soon as we find a node that is greater than one and less than another
    current = root
    stack = [current]
    lo, hi = min(p.val, q.val), max(p.val, q.val)
    while not (lo <= current.val <= hi):
        current = stack.pop()
        if p.val > current.val and q.val > current.val:
            stack.append(current.right)
        else:
            stack.append(current.left)
    return current

def LowestCommonAncestorBSTQuicker(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # as soon as we find a node that is greater than one and less than another
    lo, hi = min(p.val, q.val), max(p.val, q.val)
    while not (lo <= root.val <= hi):
        if p.val > root.val and q.val > root.val:
            root = root.right
        else:
            root = root.left
    return root

# assumes s < b
def ComputeLCA_EPI_Version(tree, s, b):
    while tree.val < s.val or tree.val > b.val: #[s, b] check if within range
        while tree.val < s.val:
            tree = tree.right 
        while tree.val > b.val:
            tree = tree.left 
    return tree


# https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/
def MaxAncestorDiff(root: Optional[TreeNode]) -> int:
    max_difference_so_far = float('-inf')

    # keep track of min and max seen as we traverse down and update the max_difference with this
    def search(node, max_above, min_above):
        nonlocal max_difference_so_far
        if node:
            new_max_above, new_min_above = max(max_above, node.val), min(min_above, node.val)
            diff = max(abs(new_max_above-node.val), abs(new_min_above-node.val))
            max_difference_so_far = max(max_difference_so_far, diff)

            search(node.left, new_max_above, new_min_above)
            search(node.right, new_max_above, new_min_above)

    search(root, root.val, root.val)
    return max_difference_so_far


# https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
def flatten(root: Optional[TreeNode]) -> None:
    current = root
    while current:
        if current.left:
            p = current.left
            while p.right:
                p = p.right
            p.right = current.right
            current.right = current.left
            current.left = None 
        current = current.right 
        

# https://leetcode.com/problems/unique-binary-search-trees/
# see DP version in Dynamic Programming section
def TotalNumberOfTrees(n: int) -> int:
    dp = {0:1, 1:1}
    
    def dfs(i):
        if i in dp:
            return dp[i]
        result = 0
        for j in range(i):
            result += dfs(j) * dfs(i-j-1)
        dp[i] = result
        return dp[i]
    
    return dfs(n)


# https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/
def FindSecondMinimumValue(root: Optional[TreeNode]) -> int:
    minimum, second_minimum = root.val, float('inf')
    queue = deque( [(root, 1)] )
    while queue:
        level = queue[0][1]
        while queue and queue[0][1] == level:
            node, _ = queue.popleft()
            if node.val != minimum and node.val < second_minimum:
                second_minimum = node.val
            if node.left: # if it has chlidren, since if it has left then it has right for this problem
                queue.append((node.left, level + 1))
                queue.append((node.right, level + 1))
    return -1 if second_minimum == float('inf') else second_minimum


# https://leetcode.com/problems/sum-of-left-leaves/
def SumOfLeftLeaves(root: Optional[TreeNode]) -> int:
    
    def traverse(node, searched_left):
        if not node:
            return 0
        elif not node.left and not node.right and searched_left:
            return node.val
        else:
            return traverse(node.left, True) + traverse(node.right, False)

    return traverse(root, False)

    
##############################################################################################
###   [GRAPHS]
##############################################################################################


# https://leetcode.com/problems/find-if-path-exists-in-graph/
def PathExists(edges: List[List[int]], source: int, destination: int) -> bool:
    if source == destination:
        return True
    
    # build bi-directional graph
    graph = {}
    for edge in edges:
        if edge[0] not in graph:
            graph[edge[0]] = []
        if edge[1] not in graph:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
        
    visited, stack = set(), [source]
    while stack:
        current = stack.pop()
        if current in graph:
            for neighbor in graph[current]:
                if neighbor == destination:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor) 
                    stack.append(neighbor)
    return False


# https://leetcode.com/problems/find-the-town-judge/
# node that has no edges out of it but every node has edge to it (sink)
def FindJudge(N: int, trust: List[List[int]]) -> int:
    counts = [0] * (N + 1)
    for edge in trust:
        src, tgt = edge[0], edge[1]
        counts[src] -= 1
        counts[tgt] += 1
    for i in range(1, len(counts)):
        if counts[i] == N - 1:
            return i
    return -1


# think DFS would have better performance on average since it visits new nodes quicker, need to investigate
def NumberOfConnectedComponents(isConnected: List[List[int]]) -> int:
    if not isConnected:
        return 0
    count = 0
    visited = set()
    for i in range(len(isConnected)):
        if i not in visited:
            q = deque([i])
            while q:
                current = q.popleft()
                for j in range(len(isConnected[current])):
                    if isConnected[current][j] == 1 and j not in visited and i != j:
                        q.append(j)
                        visited.add(j)
            visited.add(i)
            count += 1
    return count


# https://leetcode.com/problems/find-center-of-star-graph/
def CenterOfStarGraph(edges: List[List[int]]) -> int:
    n1, n2 = edges[0][0], edges[0][1]
    return n1 if n1 in edges[1] else n2


def LargestConnectedComponent(graph) -> int:
    if not graph:
        return 0
    largest = 0
    visited = set()
    for node, neighbors in graph.items():
        stack = [node]
        component_size = 1
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
                    component_size += 1
        largest = max(largest, component_size)
    return largest


# https://leetcode.com/problems/reconstruct-itinerary/
def ReconstructItinerary(tickets: List[List[str]]) -> List[str]:
    # create adjacency list
    graph = {}
    for ticket in tickets:
        departure, destination = ticket[0], ticket[1]
        if departure not in graph:
            graph[departure] = []
        if destination not in graph:
            graph[destination] = []
        graph[departure].append(destination)

    # sort the destinations lexigraphically 
    for value in graph.values():
        value.sort(reverse=True)
    
    # start with JFK in stack, at each step if there are neighbors append them
    # to the stack and if there are no neighbors then we can add it to the result
    result, stack = [], deque(["JFK"])
    while stack:
        current = stack[-1]
        if graph[current]:
            stack.append(graph[current].pop())
        else:
            result.append(stack.pop())
    return result[::-1]


# Could also do a topological sort
# https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/
def GetAllAncestorsOfAllNodes(n: int, edges: List[List[int]]) -> List[List[int]]:
    graph = {i: [] for i in range(n)}
    for source, target in edges:
        graph[source].append(target)
        
    # goal is to populate this ancestor to all children
    results = [[] for _ in range(n)]
    def dfs(ancestor, child):
        # parent can be passed down via multiple routes to child, handle this case
        if results[child] and results[child][-1] == ancestor:
            return
        if child != ancestor:
            results[child].append(ancestor)
        for subchild in graph[child]:
            dfs(ancestor, subchild)
            
    for i in range(n):
        dfs(i, i)
    return results


# https://leetcode.com/problems/all-paths-from-source-to-target/
def AllPathsFromFirstNodeToLast(graph: List[List[int]]) -> List[List[int]]:
    last_node = len(graph) - 1
    paths = []
    queue = deque([ [0, [0]] ])
    while queue:
        current, path_so_far = queue.popleft()
        if current == last_node:
            paths.append(path_so_far)
        else:
            for neighbor in graph[current]:
                queue.append([neighbor, path_so_far + [neighbor]])
    return paths


# https://leetcode.com/problems/network-delay-time/
# https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
"""
Uses Dijkstra's algorithm which does a BFS but uses a min heap to pull out 
the current shortest path, always explores shortest first, can keep a set 
of visited nodes so that we do not cycle...
"""
def NetworkDelayTime(times: List[List[int]], N: int, K: int) -> int:
    graph = dict()
    for u, v, w in times:
        graph[u] = graph.get(u, {})
        graph[v] = graph.get(v, {})
        graph[u][v] = w
    
    visited = set()
    min_heap = [(0, K)] # (total path time, node)
    while min_heap:
        current_path_time, current_node = heapq.heappop(min_heap)
        visited.add(current_node)
        if N == len(visited):
            return current_path_time
        for neighbor, time in graph[current_node].items():
            if neighbor not in visited:
                heapq.heappush(min_heap, (current_path_time + time, neighbor))
    return -1


# https://leetcode.com/problems/course-schedule-ii/
"""
Could also do a topological sort, if we encounter a cycle during it return false
Then just traverse the ordering we assemble to get the result
"""
def FindOrderOfCourses(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # store course: prereqs and preq: courses it is prereq for            
    course_to_prereqs = collections.defaultdict(set)
    prereq_to_courses = collections.defaultdict(set)
    for course, prereq in prerequisites:
        prereq_to_courses[prereq].add(course)
        course_to_prereqs[course].add(prereq)
    
    # start with the courses that have no prerequisites
    taken = []
    queue = deque([course for course in range(numCourses) if len(course_to_prereqs[course]) == 0])
    while queue:
        currently_taking = queue.popleft()
        taken.append(currently_taking)
        if len(taken) == numCourses:
            return taken
        
        # loop through the courses that this course is a prereq for
        for prereq in prereq_to_courses[currently_taking]:
            # if the course we've just taken was a prereq for the next course, remove it from its prequisites dict
            course_to_prereqs[prereq].remove(currently_taking)
            # if we've taken all of the prereqs for the new course, we'll visit it
            if len(course_to_prereqs[prereq]) == 0:
                queue.append(prereq)
    return []


# https://leetcode.com/problems/clone-graph/
def CloneGraph(node): 
    
    # populate val, but populate neighbors later
    def clone(some):
        return Node(some.val, [])
    
    # check the OG's neighbors, if we don't have that node then add it else add to its neighbors
    def dfs(some, d):
        for n in some.neighbors:
            if n not in d:
                copy = clone(n)
                d[n] = copy
                d[some].neighbors.append(copy)
                dfs(n, d)
            else:
                d[some].neighbors.append(d[n])
    
    if node:
        starter = clone(node)
        d = { node: starter }
        dfs(node, d)
        return starter


# https://leetcode.com/problems/find-eventual-safe-states/
def EventualSafeNodes(graph: List[List[int]]) -> List[int]:
    # if we find a cycle then all nodes in the cycle are not safe nodes
    n = len(graph)
    safe = {}
    result = []
    
    def dfs(node):
        if node in safe:
            return safe[node]
        # assume the node is not safe, DFS each neighbor and if neighbors are all safe we can assume node is safe 
        safe[node] = False 
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False 
        safe[node] = True
        return True
    
    for node in range(0, n):
        if dfs(node):
            result.append(node)
    return result


##############################################################################################
###   [MAPS]
##############################################################################################


# https://leetcode.com/problems/two-sum/
def TwoSumDistinct(nums, target):
    d = {value: idx for idx, value in enumerate(nums)}
    for value, idx in d.items():
        other = target - value
        if other in d:
            return [idx, d[target-value]]
    return []

def TwoSumsNotDistinct(nums, target):
    d = dict()
    for idx, value in enumerate(nums):
        if value in d:
            return [idx, d[value]]
        diff = target - value 
        d[diff] = idx
    return []


# https://leetcode.com/problems/count-the-number-of-consistent-strings/
def CountConsistentStrings(allowed: str, words: List[str]) -> int:
    lookup = set(allowed)
    result = 0
    for word in words:
        valid = True
        for char in word:
            if char not in lookup:
                valid = False
                break
        if valid:
            result += 1
    return result


# https://leetcode.com/problems/maximum-number-of-pairs-in-array/
def NumberOfPairs(nums: List[int]) -> List[int]:
    lookup = dict()
    result = [0, len(nums)]
    for num in nums:
        if lookup.get(num, 0) == 1:
            lookup[num] = 0
            result[0] += 1
            result[1] -= 2
        else:
            lookup[num] = 1
    return result


# https://leetcode.com/problems/number-of-good-pairs/
def NumIdenticalPairs(self, nums: List[int]) -> int:
    pairs = collections.defaultdict(int)
    result = 0
    for num in nums:
        result += pairs[num]
        pairs[num] += 1
    return result


# https://leetcode.com/problems/contains-duplicate-ii/
def ContainsNearbyDuplicate(nums: List[int], k: int) -> bool:
    if k > 0:        
        cache = set()
        for i, num in enumerate(nums):
            if num in cache:
                return True
            cache.add(num)
            if len(cache) == k + 1:
                cache.remove(nums[i-k])
    return False


# https://leetcode.com/problems/sum-of-unique-elements/
def SumOfUnique(nums: List[int]) -> int:
    result = 0
    counts = dict()
    for num in nums:
        if num not in counts:
            result += num
        elif counts[num] == 1:
            result -= num
        counts[num] = counts.get(num, 0) + 1
    return result
                

# https://leetcode.com/problems/check-if-number-has-equal-digit-count-and-digit-value/
def DigitCount(num: str) -> bool:
    counts = dict()
    for digit in num:
        if digit in counts:
            counts[digit] += 1
        else:
            counts[digit] = 1   
    return all(int(digit) == counts.get(str(i), 0) for i, digit in enumerate(num))   


# https://leetcode.com/problems/roman-to-integer/
def RomanToInteger(s):
    conversion = { 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000 }
    result = 0
    for i in range(len(s) - 1):
        if conversion[s[i]] < conversion[s[i+1]]:
            result -= conversion[s[i]]
        else:
            result += conversion[s[i]]
    # last one is always added
    return result + conversion[s[-1]] 


# The majority element is the element that appears more than ⌊n/2⌋ times. You may assume that the majority element always exists in the array.
def MajorityElement1(nums: List[int]) -> int:
    nums.sort()
    return nums[len(nums) // 2]

def MajorityElement2(nums):
    d = dict()
    for n in nums:
        d[n] = d[n] + 1 if n in d else 1
        if d[n] > len(nums) // 2:
            return n

# ingenious
def MajorityElement3(nums):
    candidate, count = nums[0], 0
    for n in nums:
        if candidate == n:
            count += 1
        elif count == 0:
            candidate, count = n, 1
        else:
            count -= 1
    return candidate


# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def LongestSubstringWithoutRepeatingCharactersMap(s):
    if len(s) == 0:
        return 0
    cache = dict()
    longest = 1
    for idx, letter in enumerate(s):
        if letter in s:
            longest = max(longest, len(cache))
            starting_idx = cache[letter]
            cache = {k: v for k, v in cache.items() if v > starting_idx}
        cache[letter] = idx
    return max(longest, len(cache))


# https://leetcode.com/problems/valid-sudoku/
# board doesn't have to be solved, elegant solution
def IsValidSudoko(board):
    cache = {}
    for i, row in enumerate(board):
        for j, c in enumerate(row):
            if c != '.':
                row_val = (c, i) # reverse to avoid accidental cache hits colliding with column 
                col_val = (j, c)
                square_val = (i//3, j//3, c)
                if row_val in cache or col_val in cache or square_val in cache:
                    return False 
                cache[row_val], cache[col_val], cache[square_val] = None, None, None
    return True

def IsValidSudokoBrute(board):
    dim = len(board)
    for row in board: # row
        cache = {}
        for value in row:
            if value != "." and value in cache:
                return False
            else:
                cache[value] = None
    for r in range(dim): # cols
        cache = {}
        for c in range(dim):
            if board[c][r] != "." and board[c][r] in cache:
                return False 
            else:
                cache[board[c][r]] = None 
    indices = [(x, y) for x in range(0, 9, 3) for y in range(0, 9, 3)]
    for x, y in indices: # squares
        cache = {}
        for i in range(x, x + 3):
            for j in range(y, y + 3):
                if board[i][j] != "." and board[i][j] in cache:
                    return False
                else: 
                    cache[board[i][j]] = None 
    return True


# https://leetcode.com/problems/group-anagrams/
def GroupAnagrams(strs: List[str]) -> List[List[str]]:
    result = {}
    for s in strs:
        transformed_s = ''.join(sorted(s))
        if transformed_s in result:
            result[transformed_s] += [s]
        else:
            result[transformed_s] = [s]
    return result.values()


# https://leetcode.com/problems/repeated-dna-sequences/
def FindRepeatedDnaSequences(s: str) -> List[str]:
    lookup = {}
    result = []
    for i in range(len(s)-9):
        substr = s[i:i+10]
        if substr in lookup:
            if not lookup[substr]:
                lookup[substr] = True
                result.append(substr)
        else:
            lookup[substr] = False
    return result


# https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/
# O(n) space and time
def IsPossibleDivide(nums: List[int], k: int) -> bool:
    frequencies = collections.Counter(nums)
    for num in nums:
        
        # go to the beginning of a potential sequence
        start = num
        while frequencies[start-1] > 0:
            start -= 1
        
        # fill in all sequences while there are counts available for numbers within the start and num range
        while start <= num:
            while frequencies[start] > 0:
                for victim in range(start, start + k):
                    if frequencies[victim] == 0:
                        return False 
                    frequencies[victim] -= 1
            start += 1
    return True


##############################################################################################
###   [HEAPS]
##############################################################################################

"""
1. heapify operation is actually O(n)

2. remember that heapq only applies min-heap functionality, 
   so negate all values to get a max-heap 

3. useful method  =>  heapq.nlargest(n, heap)
"""

# https://leetcode.com/problems/kth-largest-element-in-a-stream/
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.running = nums[:k]
        heapq.heapify(self.running)
        for i in range(k, len(nums)):
            heapq.heappushpop(self.running, nums[i])

    def add(self, val: int) -> int:
        heapq.heappush(self.running, val)
        if len(self.running) > self.k:
            heapq.heappop(self.running)
        return self.running[0]


# https://leetcode.com/problems/last-stone-weight/
# take 2 greatest and if they are unequal add back the difference, else destroy them
def LastStoneWeight(stones: List[int]) -> int:
    stones = [-stone for stone in stones]
    heapq.heapify(stones)
    while len(stones) >= 2:
        y, x = heapq.heappop(stones), heapq.heappop(stones)
        if x == y:
            continue
        else:
            value = y - x
        heapq.heappush(stones, value)
    return -stones[0] if stones else 0


# https://leetcode.com/problems/reorganize-string/
def ReorganizeString(s: str) -> str:
    result, counts = [], Counter(s)
    heap = [(-value, char) for char, value in counts.items()]
    heapq.heapify(heap)
    
    next_candidate = None
    if heap:
        next_candidate = heapq.heappop(heap)
        
    while heap or next_candidate:
        candidate = next_candidate
        count, char = -candidate[0], candidate[1]
        result.append(char)
        
        if not heap:
            if count - 1 == 0:
                return ''.join(result)
            else:
                return ""
        
        next_candidate = heapq.heappop(heap)
        if count - 1 > 0:
            heapq.heappush(heap, (-(count - 1), char))
    return ''.join(result)


# https://leetcode.com/problems/kth-largest-element-in-an-array/
# O(k*log(n))
def FindKthLargest(nums, k):
    a = [-n for n in nums]
    heapq.heapify(a)
    element = 0
    for _ in range(k):
        element = heapq.heappop(a)
    return -element

# NOTE see note on quickselect at top of file
def FindKthLargest_QuickSelect(nums, k):
        # use first as pivot
        pivot = random.choice(nums)

        # i think if we want kth largest distinct, could use sets instead of arrays
        less, equal, greater = [], [], []
        for num in nums:
            if num < pivot:
                less.append(num)
            elif num > pivot:
                greater.append(num)
            else:
                equal.append(num)

        # see which array the kth element lives in
        G, E = len(greater), len(equal)
        if k <= G:
            return FindKthLargest_QuickSelect(greater, k)
        elif k > G + E:
            return FindKthLargest_QuickSelect(less, k - G - E)
        else:
            return equal[0]


##############################################################################################
###   [DEPTH-FIRST SEARCH]
##############################################################################################


"""
Surrounded regions should not be on the border, which means that any 'O' on the border of the 
board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to 
an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent 
cells connected horizontally or vertically.

key is that border by default is not surrounded and any they touch will be not surrounded
    - set these to T then keep then keep them as O at the end
    - all other O's which were not reached can assume to be converted to X
"""
# https://leetcode.com/problems/surrounded-regions/
def SurroundedRegions(board):
    if not board:
        return []
    m, n = len(board), len(board[0])

    def borderDfs(i, j):
        if 0 <= i < m and 0 <= j < n and board[i][j] == "O":
            board[i][j] = "T"
            borderDfs(i+1, j)
            borderDfs(i, j+1)
            borderDfs(i-1, j)
            borderDfs(i, j-1)

    # start dfs from perimeter of the board
    for i in range(m):
        borderDfs(i, 0)     # left most
        borderDfs(i, n - 1) # right most
    for j in range(n):
        borderDfs(0, j)     # top most
        borderDfs(m - 1, j) # bottom most
    
    # set all T's to O
    for i in range(m):
        for j in range(n):
            if board[i][j] == "T":
                board[i][j] = "O"
            else:
                board[i][j] = "X"


# https://leetcode.com/problems/number-of-islands/
def NumberOfIslands(grid: List[List[str]]) -> int:
        
    def dfs(grid, i, j):
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == "1":
            grid[i][j] = "#"
            dfs(grid, i, j-1)
            dfs(grid, i-1, j)
            dfs(grid, i, j+1)
            dfs(grid, i+1, j)
            
    count = 0
    if not grid:
        return count
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "1":
                dfs(grid, i, j)
                count += 1
    return count


def CourseSchedule(intervals: List[List[int]], newInterval: List[int]):
    i, result = 0, []
    while i < len(intervals) and newInterval[0] > intervals[i][1]:
        result.append(intervals[i])
        i += 1
    
    toBeAdded = [newInterval[0], newInterval[1]]
    if i < len(intervals):
        toBeAdded[0] = min(toBeAdded[0], intervals[i][0])
        while (i < len(intervals)) and (newInterval[1] >= intervals[i][0]):
            toBeAdded[1] = max(toBeAdded[1], intervals[i][1])
            i += 1

    return result + [toBeAdded] + intervals[i:]


"""
    Could also just traverse and compare parent to child and if there is a 
    difference than return false might be a bit quicker XD
"""
def IsSingleValTree(root: Optional[TreeNode]) -> bool:
    def traverse(node, value):
        if node:
            if node.val != value:
                return False
            return traverse(node.left, value) and traverse(node.right, value)
        return True
    return traverse(root, root.val)


# https://leetcode.com/problems/n-queens/
class Solution:
    
    def SolveNQueens(self, n: int) -> List[List[str]]:
        self.boards = []
        self.board = [['.' for _ in range(n)] for _ in range(n)]
        
        def validateDiagonals(r, c):
            x = 1
            still_active = [True] * 4
            while any(still_active):
                for idx, position in enumerate([(r+x,c-x),(r+x,c+x),(r-x,c-x),(r-x,c+x)]):
                    if still_active[idx]:
                        if 0 <= position[0] < n and 0 <= position[1] < n:
                            if self.board[position[0]][position[1]] == 'Q':
                                return False
                        else:
                            still_active[idx] = False
                x += 1
            return True
        
        # we know it is not within same row, so just check column and diagonals       
        def validPlacement(r, c):
            for col in range(n):
                if col != c and self.board[r][col] == 'Q':
                    return False
            return validateDiagonals(r, c)
            
        # try backtracking DFS
        def solve(c):
            
            # base case, our column has reached the end, reset it
            if c == n:
                self.boards.append([''.join(row) for row in self.board])
                return 

            # go down each row, placing a queen
            for r in range(n):
                if self.board[r][c] == '.' and validPlacement(r, c):
                    self.board[r][c] = 'Q'
                    solve(c+1)
                    self.board[r][c] = '.'

        solve(0)
        return self.boards

# N-QUEENS using r+c, r-c lookup for diagonals
class Solution:

    def SolveNQueensV2(self, n: int) -> List[List[str]]:
        result = []
        board = [['.'] * n for _ in range(n)]

        columns = set()
        positive_diags = set() # r+c
        negative_diags = set() # r-c
        
        # we know it is not within same row, so just check column and diagonals       
        def validPlacement(r, c):
            return (c not in columns) and (r-c not in negative_diags) and (r+c not in positive_diags)
            
        # try backtracking DFS
        def solve(r):
            
            # reached the end, append the board and return
            if r == n:
                result.append([''.join(row) for row in board])
                return
            
            # loop through all columns
            for c in range(n):
                if board[r][c] != 'Q' and validPlacement(r, c):
                    
                    # update sets, then backtrack
                    columns.add(c)
                    positive_diags.add(r+c)
                    negative_diags.add(r-c)
                    board[r][c] = 'Q'
                    
                    solve(r+1)
            
                    # trigger backtracking
                    columns.remove(c)
                    positive_diags.remove(r+c)
                    negative_diags.remove(r-c)
                    board[r][c] = '.'
            
        solve(0)
        return result


# Backtracking
# https://leetcode.com/problems/course-schedule/
def CanFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    # build dictionary with course and all its prereqs
    graph = dict()
    for course, prereq in prerequisites:
        if course not in graph:
            graph[course] = [] 
        graph[course].append(prereq)

    # to detect a cycle, make sure a node is not in the stack as we process
    def cycle(node, stack):
        if node in visited:
            if node in stack:
                return True
            else:
                return False
            
        visited.add(node)
        stack.append(node)
        for neighbor in graph.get(node, []):
            if cycle(neighbor, stack):
                return True
            
        # pop it from stack once it is visited
        stack.pop()
        return False

    # possible cycles will only use node involved in the graph (since they have prereqs)
    visited = set()
    for n in graph.keys():
        if cycle(n, list()):
            return False
    return True


# Backtracking
# https://leetcode.com/problems/path-with-maximum-gold/
def GetMaximumGold(grid: List[List[int]]) -> int:
    n, m, total_gold = len(grid), len(grid[0]), 0

    def backtrack(row, col, current_gold):
        nonlocal total_gold

        # get current gold, add to current solution, update global max
        temp = grid[row][col]
        grid[row][col] = 0
        current_gold += temp
        total_gold = max(total_gold, current_gold)

        for r, c in ((row+1, col), (row-1, col), (row, col+1), (row, col-1)):
            # make sure r and c are within bounds, and only backtrack if the value is not 0
            if 0 <= r < n and 0 <= c < m and grid[r][c] != 0:
                backtrack(r, c, current_gold)

        # we set the grid value to 0 so now we need to set it back
        grid[row][col] = temp
        current_gold -= temp

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell != 0:
                backtrack(r, c, 0)
    return total_gold


# Backtracking
# https://leetcode.com/problems/combination-sum/
def CombinationSum(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    n = len(candidates)
    results = []
    
    def backtrack(i, subtotal, sequence):
        if subtotal == target:
            results.append(sequence[:])
            return
        for j in range(i, n):
            candidate = candidates[j]
            if subtotal + candidate <= target:
                sequence.append(candidate)
                backtrack(j, subtotal + candidate, sequence)
                sequence.pop()

    backtrack(0, 0, [])
    return results


# Backtracking
# https://leetcode.com/problems/combinations/
def Combine(n: int, k: int) -> List[List[int]]:
    result = []
    numbers = [i for i in range(1, n+1)]

    def backtrack(i, combination, remaining):
        if remaining == 0:
            result.append(combination[:])
            return
        for j in range(i, n):
            combination.append(numbers[j])
            backtrack(j+1, combination, remaining-1)
            combination.pop()
        
    backtrack(0, [], k)
    return result


# Backtracking
# https://leetcode.com/problems/subsets/
def Subsets(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    powerset = []

    def backtrack(i, subset):
        powerset.append(subset[:])
        for j in range(i, n):
            subset.append(nums[j])
            backtrack(j+1, subset)
            subset.pop()
            
    backtrack(0, [])
    return powerset

def Subsets_Iterative(nums: List[int]) -> List[List[int]]:
    powerset = [[]]
    for n in nums:
        powerset.extend([r + [n] for r in powerset])
    return powerset


# Backtracking
# https://leetcode.com/problems/subsets-ii/
def SubsetsWithDup(nums: List[int]) -> List[List[int]]:
    powerset = []
    nums.sort()
    n = len(nums)

    def backtrack(i, subset):
        powerset.append(subset[:])
        for j in range(i, n):
            if j > i and nums[j] == nums[j-1]:
                continue
            subset.append(nums[j])
            backtrack(j+1, subset)
            subset.pop()

    backtrack(0, [])
    return powerset

def SubsetsWithDup_Iterative(nums: List[int]) -> List[List[int]]:
        nums.sort()
        powerset, subset = [[]], []
        for i in range(len(nums)):
            # avoid generating duplicates, this check ensures we process the first dup and skip the rest
            if i > 0 and nums[i] == nums[i-1]:
                subset = [item + [nums[i]] for item in subset]
            # add value to powerset as per usual
            else:
                subset = [item + [nums[i]] for item in powerset]
            powerset += subset
        return powerset


# Backtracking
# https://leetcode.com/problems/permutations/
def AllPermutations(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    result = []
    used = [False] * n
    
    def search(path, used, result):
        if len(path) == n:
            result.append(path[:])
            return
        for i, element in enumerate(nums):
            if not used[i]:
                used[i] = True
                path.append(element)
                search(path, used, result)
                path.pop()
                used[i] = False
        
    search([], used, result)
    return result


# Backtracking
# https://leetcode.com/problems/sudoku-solver/
def SolveSudoku(board: List[List[str]]) -> None:

    def validPlacement(row, col, value):
        # check row and column
        for i in range(9):
            if board[row][i] == value or board[i][col] == value:
                return False
        # check square
        row_bound = (row // 3) * 3
        col_bound = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[i+row_bound][j+col_bound] == value:
                    return False 
        return True
    
    def solve(row, col):
        # start a new col
        if col == 9:
            col = 0
            row += 1
            # we have filled all rows
            if row == 9:
                return True
            
        if board[row][col] != '.':
            return solve(row, col+1)
        
        # if we have space available and we can place
        for value in range(1, 10):
            if validPlacement(row, col, str(value)):
                board[row][col] = str(value)
                if solve(row, col+1):
                    return True
        # backtracking
        board[row][col] = '.'
        return False
    
    solve(0, 0)


##############################################################################################
###   [BREADTH-FIRST SEARCH]
##############################################################################################


def MaximumDepthOfBinaryTreeBFS(root: Optional[TreeNode]) -> int:
    def helper(node, count):
        if node:
            return max(helper(node.left, count + 1), helper(node.right, count + 1))
        return count
    return helper(root, 0)

"""
    Could also run this using BFS with queue going level by level and 
    first node that is reached without a child, the level is returned
"""
def MinimumDepthOfBinaryTree(root):
    def traverse(node, depth):
        if node.left and node.right:
            return min(traverse(node.left, depth + 1), traverse(node.right, depth + 1))
        elif node.left:
            return traverse(node.left, depth + 1)
        elif node.right:
            return traverse(node.right, depth + 1)
        else:
            return depth
    if not root:
        return 0
    return traverse(root, 1)

def MinimumDepthOfBinaryTreeConcise(root):
    if not root:
        return 0
    if not root.left or not root.right:
        return MinimumDepthOfBinaryTreeConcise(root.left) + MinimumDepthOfBinaryTreeConcise(root.right) + 1
    return min(MinimumDepthOfBinaryTreeConcise(root.left), MinimumDepthOfBinaryTreeConcise(root.right)) + 1 


def MaxDepthNAryTree(root):
    if not root:
        return 0

    level, max_depth = 1, 1
    q = deque([(root, 1)])
    while q:
        while q and q[-1][1] == level:
            node, depth = q.popleft()
            if len(node.children) == 0:
                max_depth = max(max_depth, depth)
            for child in node.children:
                q.append((child, depth + 1))
        level += 1
    return max_depth


# https://leetcode.com/problems/number-of-closed-islands/
def ClosedIslandsCount(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    
    def valid(x, y):
        return 0 <= x < m and 0 <= y < n
    def dfs(x, y):
        if valid(x, y) and grid[x][y] == 0:
            grid[x][y] = -1
            if x == 0 or x == m-1 or y == 0 or y == n-1:
                self.valid_grid = False
                return
            dfs(x+1,y)
            dfs(x,y+1)
            dfs(x-1,y)
            dfs(x,y-1)
        
    # start DFS from inner grid and increase count if island
    count = 0
    for x in range(1, m-1):
        for y in range(1, n-1):
            if grid[x][y] == 0:
                self.valid_grid = True
                dfs(x, y)
                if self.valid_grid:
                    count += 1
    return count
        

# https://leetcode.com/problems/max-area-of-island/
def MaximumAreaOfIsland(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    
    def dfs(x, y):
        if 0 <= x < m and 0 <= y < n and grid[x][y]:
            grid[x][y] = 0
            self.current_max_area += 1
            dfs(x+1,y)
            dfs(x,y+1)
            dfs(x-1,y)
            dfs(x,y-1)
    
    best_max_area = 0
    for x in range(m):
        for y in range(n):
            if grid[x][y]:
                self.current_max_area = 0
                dfs(x, y)
                best_max_area = max(best_max_area, self.current_max_area)
    return best_max_area


# https://leetcode.com/problems/shortest-bridge/
def ShortestBridge(grid: List[List[int]]) -> int:
    n = len(grid)
    def valid(r, c):
        return 0 <= r < n and 0 <= c < n
    def next_coordinates(r, c):
        return ((r+1, c), (r, c+1), (r-1, c), (r, c-1))
    
    # used to find all coordinates in the first island
    visited = set()
    def dfs(r, c):
        if valid(r, c) and grid[r][c] == 1 and (r, c) not in visited: 
            visited.add((r, c))
            for row, col in next_coordinates(r, c): 
                dfs(row, col)
        
    # expand out one level each time, first to reach a 1 we return 
    def bfs():
        expansions = 0
        queue = deque(visited)
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for row, col in next_coordinates(r, c):
                    if valid(row, col) and (row, col) not in visited:
                        if grid[row][col] == 1:
                            return expansions
                        queue.append((row, col))
                        visited.add((row, col))
            expansions += 1
        return expansions
    
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                dfs(r, c)
                return bfs()
    return -1


##############################################################################################
###   EPI SEARCH
##############################################################################################


# could also use hashmap
def IntersectionTwoSortedArrays(nums1: List[int], nums2: List[int]) -> List[int]:
    nums1.sort()
    nums2.sort()
    n1, n2, result = 0, 0, []
    while n1 < len(nums1) and n2 < len(nums2):
        if nums1[n1] == nums2[n2]:
            if len(result) == 0 or result[-1] != nums1[n1]:
                result.append(nums1[n1])
            n1, n2 = n1 + 1, n2 + 1
        elif nums1[n1] > nums2[n2]:
            n2 += 1
        else:
            n1 += 1
    return result


# https://leetcode.com/problems/merge-sorted-array/
# assume A1 has enough room to put elements of A2
def MergeTwoSortedArrays(A1, A2):
    m, n = len(A1), len(A2) 
    write_idx = m + n - 1
    m -= 1
    n -= 1

    while n >= 0:
        if m >= 0 and A2[n] < A1[m]: 
            A1[write_idx] = A1[m]
            m -= 1
        else: # A2[n] > A1[m] or A2[n] == A1[m]
            A1[write_idx] = A2[n]
            n -= 1
        write_idx -= 1


# https://leetcode.com/problems/h-index/
def HIndex(citations: List[int]) -> int:
    n = len(citations)
    citations.sort()
    for i, citation in enumerate(citations):
        if citation >= n - i:
            return n - i
    return 0


##############################################################################################
###   [DYNAMIC PROGRAMMING]
##############################################################################################


""" NOTES
typically want to use @lru_cache(None) to cache recursive results when they are likely to come 
up again with same arguements, None means cache can grow without any limitations. no max size

useful for fibonacci can set lru_cache(2) so it will always cache past two results
"""


# https://leetcode.com/problems/maximum-subarray/
def MaximumSubarray(nums):
    dp = [0] * len(nums)
    dp[0] = nums[0]
    running_max = dp[0]
    for i in range(1, len(nums)):
        dp[i] = max(nums[i], dp[i-1] + nums[i])
        running_max = max(running_max, dp[i])
    return running_max

def MaximumSubarrayV2(nums):
    max_seen = max_end = float('-inf')
    for num in nums:
        max_end = max(num, num + max_end)
        max_seen = max(max_seen, max_end)
    return max_seen


# https://leetcode.com/problems/n-th-tribonacci-number/
def Tribonacci(n: int) -> int:
    f, s, t = 0, 1, 1
    if n == 0:
        return f
    if n < 3:
        return s
    for _ in range(3, n + 1):
        next_fib = f + s + t
        f, s, t = s, t, next_fib
    return next_fib


# https://leetcode.com/problems/climbing-stairs/
# Can reduce memory to constant by using two variables instead of a whole array
def ClimbStairs(n: int) -> int:
    best_fst, best_scd = 1, 2
    if n == 1:
        return 1
    if n == 2:
        return 2
    for _ in range(2, n):
        best_fst, best_scd = best_scd, best_fst + best_scd
    return best_scd


def GeneratePascalsTriangle(numRows: int) -> List[List[int]]:
    result = [[1] for _ in range(numRows)]
    for i in range(1, numRows):
        for j in range(1, i + 1):
            l = r = 0
            if j-1 >= 0:
                l = result[i-1][j-1]
            if j < i:
                r = result[i-1][j]
            result[i].append(l + r)
    return result


# https://leetcode.com/problems/interleaving-string/
def CanInterleaveString(s1: str, s2: str, s3: str) -> bool:
    m, n = len(s1), len(s2)
    if m + n != len(s3): 
        return False
    
    @lru_cache(None)
    def dfs_sp(i, j):
        if i == m and j == n: 
            return True 
        if i < m and s1[i] == s3[i+j]:  # Case match s1[i] with s3[i+j]
            result = dfs_sp(i+1, j)
            if result:
                return True 
        if j < n and s2[j] == s3[i+j]:  # Case match s2[j] with s3[i+j]
            result = dfs_sp(i, j+1)
            if result:
                return True
        return False

    return dfs_sp(0, 0)


# https://leetcode.com/problems/count-number-of-teams/
def CountNumberOfTeams(rating: List[int]) -> int:
    # O(n^2)
    # create a list of all that are greater / less than element at rating[i]
    # basically treating each number (i in range(n)) as the lower
    
    n = len(rating)
    greater, less = {}, {}
    for i in range(n):
        for j in range(i+1, n):
            if rating[j] > rating[i]:
                greater[i] = greater.get(i, 0) + 1
            else: # rating[j] < rating[i]
                less[i] = less.get(i, 0) + 1

    # basically take i then take j then all others that are less than j or greater than j get added
    result = 0
    for i in range(n - 2):
        for j in range(i+1, n):
            if rating[j] > rating[i]:
                result += greater.get(j, 0)
            else:
                result += less.get(j, 0)
    
    return result


# https://leetcode.com/problems/house-robber/
def HouseRobber(nums: List[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    
    array = [nums[0], max(nums[0], nums[1])]
    for i in range(2, len(nums)):
        num = nums[i]
        array.append(max(array[i-2] + num, array[i-1]))
    return array[-1]


# https://leetcode.com/problems/house-robber-ii/
def HouseRobberII(nums: List[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    def helper(offset):
        offset_add =  0 if not offset else 1
        offset_sub = -1 if not offset else 0
        rob_one, rob_two = nums[offset_add], max(nums[offset_add], nums[offset_add+1])
        for i in range(offset_add + 2, len(nums) + offset_sub):
            temp_rob = max(nums[i] + rob_one, rob_two)
            rob_one, rob_two = rob_two, temp_rob
        return rob_two
    
    return max(helper(True), helper(False))


# https://leetcode.com/problems/maximum-alternating-subsequence-sum/
def MaxAlternatingSum(nums: List[int]) -> int:
    # add first, subtract second, add third, etc
    # sum even is if first subsequence added, sum odd is if it's subtracted
    sum_even, sum_odd = 0, 0
    
    for num in nums:
        temp_even = max(sum_odd + num, sum_even)
        temp_odd = max(sum_even - num, sum_odd)
        sum_even, sum_odd = temp_even, temp_odd
                        
    # first value we are adding is always added
    return sum_even


# https://leetcode.com/problems/target-sum/
def FindTargetSumWays(nums: List[int], target: int) -> int:
    dp = {} # (index, total) -> number of ways to get to target value
    
    def backtracking(i, current_total):
        if i == len(nums):
            return 1 if current_total == target else 0
        if (i, current_total) in dp:
            return dp[(i, current_total)]
        
        dp[(i, current_total)] = backtracking(i+1, current_total + nums[i]) + backtracking(i+1, current_total - nums[i])
        return dp[(i, current_total)]
    
    return backtracking(0, 0)


# https://leetcode.com/problems/partition-equal-subset-sum/
# O(2^n) is brute force because at each step we have two choces (could do DFS with backtracking)
# here we reduce time to sum(nums)
def CanPartition(nums: List[int]) -> bool:
    total = sum(nums)
    if total % 2 == 1:
        return False
    half = total // 2 
    
    all_sums = set([0])
    for num in nums:
        if num == half:
            return True
        next_all_sums = set()
        for sums in all_sums:
            new_sum = sums + num
            if new_sum == half:
                return True
            next_all_sums.add(new_sum)
            next_all_sums.add(sums)
        all_sums = next_all_sums
    return False


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
def MaximumProfit(prices: List[int]) -> int:
    # state:
    # buy -> i + 1 (True)
    # sell -> i + 2 (False)
    # O(2*n) runtime and space
    dp = {} # k: (index, buy/sell) v: max_profit

    def dfs(i, buying):
        if i >= len(prices):
            return 0 # no profit here
        if (i, buying) in dp:
            return dp[(i, buying)] # ax profit for this key is already stored
        
        # now we can choose to buy or sell or cooldown
        cooldown = dfs(i+1, buying)
        if buying:
            buy = dfs(i+1, not buying) - prices[i]
            dp[(i, buying)] = max(buy, cooldown)
        else:
            sell = dfs(i+2, not buying) + prices[i]
            dp[(i, buying)] = max(sell, cooldown)
        return dp[(i, buying)]
            
    return dfs(0, True)


# https://leetcode.com/problems/coin-change/
def CoinChange(coins: List[int], amount: int) -> int:
    dp = {}
    coins.sort()
    
    def dfs(remaining):
        if remaining == 0: 
            return 0
        if remaining in dp:
            return dp[remaining]
        
        coins_used = float('inf')
        for coin in coins:
            if remaining-coin >= 0:
                coins_used = min(coins_used, dfs(remaining-coin) + 1)
            else:
                break
        dp[remaining] = coins_used
        return dp[remaining]
    
    result = dfs(amount)
    return result if result != float('inf') else -1        


# https://leetcode.com/problems/coin-change-2/
# O(n*m) is time and space complexity
def CoinChangeII_DFS_With_Cache(amount: int, coins: List[int]) -> int:
    dp = {}
    def dfs(i, a):
        if a == amount:
            return 1
        if a > amount or i == len(coins):
            return 0
        if (i, a) in dp:
            return dp[(i, a)]
        
        dp[(i, a)]  = dfs(i, a + coins[i]) + dfs(i+1, a)
        return dp[(i, a)]
    return dfs(0, 0)

# https://leetcode.com/problems/coin-change-2/
# O(n*m) is time and space complexity
def CoinChangeII_DP(amount: int, coins: List[int]) -> int:
    n = len(coins)
    dp = [[0] * (amount + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 1
    
    for i in range(n - 1, -1, -1):
        for j in range(1, amount + 1):
            bottom = dp[i+1][j]
            left = dp[i][j-coins[i]] if j-coins[i] >= 0 else 0
            dp[i][j] = bottom + left 
            
    return dp[0][-1]
            
# O(n*m) but with O(n) space
def CoinChangeII_DP_LessSpace(amount: int, coins: List[int]) -> int:
    dp = [1] + [0] * (amount)
    for coin in coins:
            for i in range(1, amount + 1):
                if i - coin >= 0:
                        dp[i] += dp[i - coin]
    return dp[-1]


# https://leetcode.com/problems/minimum-cost-for-tickets/
def MinimumCostTickets(days: List[int], costs: List[int]) -> int:
    dp = {} # key is index, value is lowest cost
    
    def dfs(i):
        if i == len(days):
            return 0
        if i in dp:
            return dp[i]
        lowest = float('inf')
        for cost, day in zip(costs, [1, 7, 30]):
            further_days = days[i] + day
            j = i
            while j < len(days) and days[j] < further_days:
                j += 1
            lowest = min(lowest, dfs(j) + cost)
        dp[i] = lowest
        return dp[i]

    return dfs(0) 


# https://leetcode.com/problems/unique-binary-search-trees/
# consider n=6 aka [1,2,3,4,5,6] looping through j pick each as root and compute number of search trees
# i.e. root=3
#       [1, 2, `3`, 4, 5, 6]
#
def NumOfBinarySearchTrees_DP(n: int) -> int:
    trees = [0] * (n+1)
    trees[0] = 1
    trees[1] = 1
    for root in range(2, n+1):
        for j in range(root):
            trees[root] += trees[j] * trees[root-j-1]
    return trees[n]

def NumOfBinarySearchTrees_DFS(n: int) -> int:
    # binary tree of 0 and 1 nodes
    dp = {0:1, 1:1}
    
    def dfs(i):
        if i in dp:
            return dp[i]
        total = 0
        for root in range(i):
            left = dfs(root)
            right = dfs(i-root-1)
            total += left * right
        dp[root] = total
        return total
    
    return dfs(n)


# https://leetcode.com/problems/min-cost-climbing-stairs/
def MinCostClimbingStairs_DP(cost: List[int]) -> int:
    n = len(cost)
    cost.append(0)
    
    dp = [0] * (n + 1)
    dp[0] = cost[0]
    dp[1] = cost[1]
    
    for i in range(2, n + 1):
        dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
    return dp[-1]

def MinCostClimbingStairs_DFS(cost: List[int]) -> int:
    n = len(cost)
    dp = {}
    
    def dfs(i):
        if i >= n:
            return 0
        if i in dp:
            return dp[i]

        dp[i] = min(dfs(i+1) + cost[i], dfs(i+2) + cost[i])
        return dp[i]
        
    return min(dfs(0), dfs(1))


# https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/
def NumOfSubarraysThatSumToOddNumbers(arr: List[int]) -> int:
    # let odd[i] store the current number of odd sums ending at A[i]
    # let odd[i-1] store the number of all previous subarrays whos sum is odd
    # let even[i] store the current number of even sums ending at A[i]
    # let even[i-1] store the number of all previous subarrays whos sum is even
    evens, odds = [0] * len(arr), [0] * len(arr)
    odds[0] = arr[0] % 2
    evens[0] = 1 - odds[0]
    
    total_odd = odds[0]
    for i in range(1, len(arr)):
        n = arr[i]
        if n % 2 == 1: # odd
            # all the PREVIOUS subarrays that had an ODD sum become even 
            # all the PREVIOUS subarrays that had an EVEN sum become odd, plus itself (arr[i] is odd remember)
            evens[i] = odds[i-1]
            odds[i] = evens[i-1] + 1
        else: # even
            # all the PREVIOUS subarrays that had an ODD sum STAY odd 
            # all the PREVIOUS subarrays that had an EVEN sum STAY even, plus itself (arr[i] is even remember)
            evens[i] = evens[i-1] + 1
            odds[i] = odds[i-1]
        total_odd += odds[i]    
    
    return total_odd % 1000000007 # inherent to leetcode solution


# https://leetcode.com/problems/delete-and-earn/
def DeleteAndEarn(nums: List[int]) -> int:
    # take a count of each number so we can get its sum
    counts = collections.Counter(nums)
    n = max(counts.keys()) + 1
    
    # total number of elements we have to iterate through is up to the max in the input
    dp = [0] * n
    dp[1] = counts.get(1, 0)
    
    for i in range(2, n):
        # max of the previous if we choose to skip, and taking the number two places before and adding sum
        sum_of_new_number = counts.get(i, 0) * i 
        dp[i] = max(dp[i-1], dp[i-2] + sum_of_new_number)
    return dp[-1]


# https://leetcode.com/problems/maximum-product-subarray/
def MaxProduct(nums: List[int]) -> int:
    n = len(nums)
    positives = [0] * (n+1)
    negatives = [0] * (n+1)
    positives[1] = max(0, nums[0])
    negatives[1] = min(0, nums[0])
    result = nums[0]
    
    for i in range(1, n):
        value = nums[i]
        if value > 0:
            positives[i+1] = max(value, positives[i] * value) # stays positive, so safe to multiply
            negatives[i+1] = min(value, negatives[i] * value) # stays negative, so safe to multiply
        elif value < 0:
            positives[i+1] = max(value, negatives[i] * value) # largest negative becomes positive
            negatives[i+1] = min(value, positives[i] * value) # largest positive becomes negative
        else:
            negatives[i+1] = positives[i+1] = 0
        result = max(result, positives[i+1])
        
    return result

def MaxProduct_ConstantSpace(nums: List[int]) -> int:
    positive = max(0, nums[0])
    negative = min(0, nums[0])
    result = nums[0]
    
    for i in range(1, len(nums)):
        value = nums[i]
        if value > 0:
            positive, negative = max(value, positive*value), min(value, negative*value)
        elif value < 0:
            positive, negative = max(value, negative*value), min(value, positive*value)
        else:
            positive, negative = 0, 0
        result = max(result, positive)   
    
    return result


# https://leetcode.com/problems/longest-common-subsequence/   
def LongestCommonSubsequence(text1: str, text2: str) -> int:
    dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    for i in range(len(text1) - 1, -1, -1):
        for j in range(len(text2) - 1, -1, -1):
            if text1[i] == text2[j]:
                dp[i][j] = dp[i+1][j+1] + 1
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]


# O(n^2)
# https://leetcode.com/problems/longest-increasing-subsequence/        
def LengthOfLIS(nums: List[int]) -> int:
    dp = [1] * len(nums)
    maximum = 1
    for i in range(len(nums) - 1, -1, -1):
        for j in range(i+1, len(nums)):
            if nums[j] > nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
        maximum = max(maximum, dp[i])
    return maximum

# O (n logn) 
def LengthOfLIS_Optimal(nums: List[int]) -> int:
    a = []
    for num in nums:
        idx = bisect.bisect_left(a, num)
        if idx == len(a):
            a.append(num)
        else:
            a[idx] = num
    return len(a)


# https://leetcode.com/problems/edit-distance/
def MinEditDistance(word1: str, word2: str) -> int:
    #      word2 j
    #    a  c  d  ''
    # a  .  .  .  3
    # c  .  .  .  2
    # e  .  .  .  1
    # '' 3  2  1  0
    w1, w2 = len(word1), len(word2)
    dp = [[0 for _ in range(w2+1)] for _ in range(w1)]
    dp.append([i for i in range(w2, -1, -1)])
    for i in range(1, w1+1):
        dp[~i][-1] = i

    for i in range(w1-1, -1, -1):
        for j in range(w2-1, -1, -1):
            
            # if both chars are the same, advance pointers
            if word1[i] == word2[j]:
                dp[i][j] += dp[i+1][j+1]
            
            # both operations cost one
            else:
                #          down (remove)     right (add)      diag (replace)
                dp[i][j] = min(dp[i+1][j], dp[i][j+1], dp[i+1][j+1]) + 1
                
    return dp[0][0]


def NinEditDistance_TopDown(s1: str, s2: str) -> int:
    @lru_cache(None)
    def dp(i, j):
        if i == 0: 
            return j  # Need to insert j chars
        if j == 0: 
            return i  # Need to delete i chars
        if s1[i-1] == s2[j-1]:
            return dp(i-1, j-1)
        return min(dp(i-1, j), dp(i, j-1), dp(i-1, j-1)) + 1
    return dp(len(s1), len(s2))


# https://leetcode.com/problems/integer-break/
def IntegerBreak_DFS(n: int) -> int:
    dp = { 1:1 }
    def dfs(num):
        if num in dp:
            return dp[num]
        dp[num] = 0 if num == n else num
        for i in range(1, num):
            product = dfs(i) * dfs(num-i)
            dp[num] = max(dp[num], product)
        return dp[num]
    return dfs(n)

def IntegerBreak_DP(n: int) -> int:
    dp = { 1:1 }
    for num in range(2, n + 1):
        dp[num] = 0 if num == n else num
        for i in range(1, num):
            dp[num] = max(dp[num], dp[i] * dp[num-i])
    return dp[n]


# https://leetcode.com/problems/stone-game/
def StoneGameV1(piles: List[int]) -> bool:
    dp = {}
    
    def dfs(l, r):
        if l > r:
            return 0
        if (l, r) in dp:
            return dp[(l, r)]
        
        alices_choice = ((r - l) % 2 == 1)
        take_left = piles[l] if alices_choice else 0 # bobs_choice
        take_right = piles[r] if alices_choice else 0 # bobs_choice
        dp[(l, r)] = max(dfs(l+1, r) + take_left, dfs(l, r-1) + take_right)
        return dp[(l, r)]
    
    half = sum(piles) // 2
    return dfs(0, len(piles) - 1) > half

# if Alice plays optimally, she will always win
def StoneGameV1(piles: List[int]) -> bool:
    return True


##############################################################################################
###   [MISC]
##############################################################################################


# https://leetcode.com/problems/number-of-1-bits/
def HammingWeight(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


# https://leetcode.com/problems/flipping-an-image/
def FlipAndInvertImage(image: List[List[int]]) -> List[List[int]]:
    for row in image:
        l, r = 0, len(row)-1
        while l <= r:
            row[l], row[r] = row[r] ^ 1, row[l] ^ 1
            l += 1
            r -= 1
    return image


# https://leetcode.com/problems/reverse-integer/
def ReverseInteger(x: int) -> int:
    new_digit = 0
    curr = abs(x)
    if x != 0:
        sign = int(curr / x)
    else:
        sign = 1
    while curr > 0:
        rem = curr % 10
        new_digit = 10*new_digit + rem
        curr = curr // 10
    if -2**31 <= sign*new_digit <= 2**31 - 1:
        return sign*new_digit
    return 0

# https://leetcode.com/problems/reverse-bits/
def ReverseBits(n: int) -> int:
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# https://leetcode.com/problems/distribute-candies-to-people/
def DistributeCandies(candies: int, num_people: int) -> List[int]:
    result = [0] * num_people
    i = 0
    while candies > 0:
        remaining = min(candies, i + 1)
        result[i % num_people] += remaining
        candies -= i + 1
        i += 1
    return result


# https://leetcode.com/problems/add-digits/
def AddDigitsMathematical(num):
    return 0 if num == 0 else (num - 1) % 9 + 1

def AddDigitsLoop(num):
    while num > 9:
        num = sum(int(d) for d in str(num))
    return num


# https://leetcode.com/problems/generate-parentheses/
def GenerateParenthesis(n: int) -> List[str]:
    result = []

    def helper(n_open: int, n_close: int, agg: str, result: List[int]):
        if n_open == 0 and n_close == 0:
            result.append(agg)
            return
        # if open ( is available then add one and recurse
        if n_open > 0:
            helper(n_open-1, n_close, agg + "(", result)
        # if there are extra ) available, use one
        if n_close > n_open:
            helper(n_open, n_close-1, agg + ")", result)
            
    helper(n, n, "", result)
    return result


# https://leetcode.com/problems/happy-number/
"""
Write an algorithm to determine if a number n is happy.
A happy number is a number defined by the following process:
    Starting with any positive integer, replace the number by the sum of the squares of its digits.
    Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
    Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.

To see why it converges take a look at example of 2 that infinitely goes...
                   |-----------------------------------|-------------- ...
2 -> 4 -> 8 -> 16 -> 37 -> 30 -> 9 -> 81 -> 65 ->  61 -> 37 -> 30 -> 9 ...
"""
def IsHappy(n: int) -> bool:
    # tortoise and hair method
    def square(val):
        new_n = 0
        while val > 0:
            digit = val % 10
            new_n += digit**2
            val //= 10
        return new_n
    slow, fast = square(n), square(square(n))
    while slow != fast and fast != 1:
        slow, fast = square(slow), square(square(fast))
    return fast == 1


# https://leetcode.com/problems/single-element-in-a-sorted-array/
# all other numbers appear twice, use XOR and result is one that appears once
# works for sorted or unsorted, for sorted we always know two same values are next 
# to each other so can check for that 
def FindOnlyNonDuplicateInSortedArray(nums: List[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result


# https://leetcode.com/problems/excel-sheet-column-title/
# the character of the remainder each time is the last character
def ConvertNumberToTitle(columnNumber: int) -> str:
    result = []
    base = ord('A')
    while columnNumber > 0:
        remainder = (columnNumber - 1) % 26
        columnNumber = (columnNumber - 1) // 26
        result.append(chr(base + remainder))
    return ''.join(reversed(result))


# https://leetcode.com/problems/excel-sheet-column-number/
def ConvertTitleToNumber(columnTitle: str) -> int:
    result, incr = 0, 1 
    base = ord('A')
    for c in reversed(columnTitle):
        result += (ord(c) + 1 - base) * incr
        incr *= 26 
    return result


# https://leetcode.com/problems/lru-cache/
# can also solve this problem by using doubly linked list with cache
# the linked list helps with eviction and reordering 
from collections import OrderedDict # uses doubly linked list under the hood

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1
    
    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# https://leetcode.com/problems/first-bad-version/
def firstBadVersion(n: int) -> int:
    
    # implemented by LeetCode
    def isBadVersion(version):
        pass

    @lru_cache(None)
    def search(lo, hi):
        if lo > hi:
            return lo
        mid = lo + (hi-lo) // 2
        bad_version = isBadVersion(mid)
        if bad_version:
            return search(lo, mid-1)
        else:
            return search(mid+1, hi)
        
    return search(0, n-1)


# https://leetcode.com/problems/validate-ip-address/
def ValidIPAddress(IP):
    def IPv4(IP):
        def starts_with(s, match):
            return s.startswith(match)

        tokens = IP.split(".")
        if len(tokens) != 4:
            return "Neither"
        try:
            for token in tokens:
                if starts_with(token, "0") and len(token) > 1:
                    return "Neither"
                if not token.isdigit():
                    return "Neither"
                res = int(token)
                if res < 0 or res > 255:
                    return "Neither"
        except Exception as e:
            return "Neither"
        return "IPv4"
    
    def IPv6(IP):
        def good_hex(hex_):
            match = re.match("^[A-Fa-f0-9]+$", hex_)
            return True if match else False
                
        tokens = IP.split(":")
        if len(tokens) != 8:
            return "Neither"
        for i in tokens:
            size = len(i)
            if size < 1 or size > 4:
                return "Neither"
            if not good_hex(i):
                return "Neither"
        return "IPv6"
    
    return_one = IPv4(IP)
    if return_one != "Neither":
        return return_one
    else:
        return IPv6(IP)


# https://leetcode.com/problems/powx-n/
def MyPowV1(x, n):
    def helper(orig, x, n):
        an = abs(n)
        if an == 0:
            return 1.0
        if an == 1:
            return x
        new_n = n-1 if n > 1 else n+1
        return helper(orig, orig*x, new_n)

    result = helper(x, x, n)
    if n != abs(n):
        return 1.0 / result
    else:
        return result

def MyPowV2(x, n):
    if x == 0:
        return 1.0
    result = 1.0
    for i in range(0, abs(n)):
        result *= x
    if n != abs(n):
        return 1.0 / result
    else:
        return result


# https://leetcode.com/problems/power-of-two/
"""
bit manipulation problem for n & (n-1) trick, which removes the last non-zero bit from our number
example:
    1. n = 100000, n-1 = 011111 and n & (n-1) = 000000, so if it is power of two, result is zero
    2. n = 101110, n-1 = 101101 and n & (n-1) = 101100, number is not power of two and result is not zero
"""
def IsPowerOfTwo(n: int) -> bool:
    return n > 0 and not (n & n-1)


# https://leetcode.com/problems/add-two-integers/
# they cant be fr here LOL ...  
def Sum(num1: int, num2: int) -> int:
    return num1 + num2


# https://leetcode.com/problems/minimum-sum-of-four-digit-number-after-splitting-digits/
def MinimumSum(num: int) -> int:
    n = list(str(num))
    n.sort()
    return int(n[0] + n[2]) + int(n[1] + n[3])


##############################################################################################
###   [GREEDY & INVARIANTS]
##############################################################################################

""" NOTES 
1. A greedy algo is often right choice for optimization problems where there's a natural set of choices to choose from
2. Often easier to conceptualize recursively, then implement iteratively for better performance
3. Even if greedy does not give optimal results, can give insights into optimal soln or serve as heuristic
"""

# https://leetcode.com/problems/assign-cookies/
def GiveCookieToGreedyChildren(g: List[int], s: List[int]) -> int:
    g.sort(reverse=True)
    s.sort(reverse=True)
    i = j = 0
    while i < len(g) and j < len(s):
        # if we can, give the largest available cookie to largest greedy factor
        if g[i] <= s[j]:
            j += 1
        i += 1
    return j


# https://leetcode.com/problems/can-place-flowers/
def CanPlaceFlowers(flowerbed: List[int], n: int) -> bool:
    # handle edges initially
    if flowerbed[0] == 0 and (len(flowerbed) == 1 or (len(flowerbed) > 1 and flowerbed[1] == 0)):
        flowerbed[0] = 1
        n -= 1
    if len(flowerbed) > 1 and flowerbed[-1] == 0 and flowerbed[-2] == 0:
        flowerbed[-1] = 1
        n -= 1
        
    for i in range(1, len(flowerbed) - 1):
        if flowerbed[i] == 0 and flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
            flowerbed[i] = 1
            n -= 1
        if n == 0:
            return True
            
    return n <= 0


# https://leetcode.com/problems/lemonade-change/
def LemonadeChange(bills: List[int]) -> bool:
    num5 = num10 = 0
    for bill in bills:
        if bill == 5:
            num5 += 1
        elif bill == 10:
            if not num5:
                return False 
            num5 -= 1
            num10 += 1
        else:
            if num10:
                bill -= 10
                num10 -= 1
            if bill == 10 and num5:
                num5 -= 1
            elif bill == 20 and num5 > 2:
                num5 -= 3
            else:
                return False 
    return True


# 2n or n^2 denominations (only ones solvable with greedy)
def CoinChangeUS(n):
    coins = [100, 50, 25, 10, 5, 1]
    num_coins = 0
    for coin in coins:
        num_coins += n // coin 
        n %= coin
    return num_coins
    

"""
    Could also solve this with hashtable in O(n) but that requires O(n) extra space, assumes sorted
"""
def HasTwoSum(A, target):
    start, end = 0, len(A) - 1
    while start <= end:
        if A[start] + A[end] == target:
            return True
        elif A[start] + A[end] > target:
            end -= 1
        else: # A[start] + A[end] < target
            start += 1
    return False


def HasThreeSum(A, target):
    A.sort()
    return any(HasTwoSum(A, target - a) for a in A)


# https://leetcode.com/problems/letter-combinations-of-a-phone-number/
def LetterCombinations(digits: str) -> List[str]:
    digits_map = {
        "2":  ["a", "b", "c"],
        "3":  ["d", "e", "f"],
        "4":  ["g", "h", "i"],
        "5":  ["j", "k", "l"],
        "6":  ["m", "n", "o"],
        "7":  ["p", "q", "r", "s"],
        "8":  ["t", "u", "v"],
        "9":  ["w", "x", "y", "z"]    
    }
    
    def helper(idx, aggregator, curr):
        if idx == len(digits):
            aggregator.append(curr)
        else:
            for c in digits_map[digits[idx]]:
                helper(idx+1, aggregator, curr+c)

    agg = []
    if len(digits) > 0:
        helper(0, agg, "") # populate agg using reference
        return agg
    else:
        return agg
        

# https://leetcode.com/problems/boats-to-save-people/
def NumRescueBoats(people: List[int], limit: int) -> int:
    people.sort()
    l, r = 0, len(people) - 1
    boats_used = 0
    while l <= r:
        if people[l] + people[r] <= limit:
            boats_used += 1
            l += 1
            r -= 1
        else: 
            boats_used += 1
            r -= 1
    return boats_used


##############################################################################################
###   [SORTING]
##############################################################################################


# https://leetcode.com/problems/sort-colors/
# this solution uses quicksort
def SortColors(nums: List[int]) -> None:
    
    def partition(lo, hi):
        # set pivot to the last value, and j to the first
        pivot, j = nums[hi], lo
        for i in range(lo, hi):
            if nums[i] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
        # swap the pivot with the j location
        nums[j], nums[hi] = nums[hi], nums[j]
        return j
    
    def quicksort(lo, hi):
        if len(nums) == 1:
            return nums
        
        if lo < hi:
            j = partition(lo, hi)
            quicksort(lo, j - 1)
            quicksort(j + 1, hi)
        
    quicksort(0, len(nums) - 1)

# takes advantage of nums only having 0,1,2 as values
def SortColorsOnePass(nums: List[int]) -> None:
    if len(nums) == 1:
        return nums
    
    beg, mid, end = 0, 0, len(nums) - 1
    while mid <= end:
        # will never have a 2 behind mid, so can always advance mid
        if nums[mid] == 0:
            nums[beg], nums[mid] = nums[mid], nums[beg]
            beg += 1
            mid += 1
        elif nums[mid] == 2:
            nums[mid], nums[end] = nums[end], nums[mid]
            end -= 1
        else: #nums[mid] == 1
            mid += 1


# https://leetcode.com/problems/sort-characters-by-frequency/
def SortCharsByFrequency(s: str) -> str:
    n = len(s)
    counts = Counter(s)
    buckets = [[] for _ in range(n+1)]
    for char, count in counts.items():
        buckets[count].append(char)
    result = []
    for i in range(n, 0, -1):
        for char in buckets[i]:
            result.append(char * i)
    return ''.join(result)


# https://leetcode.com/problems/sort-array-by-increasing-frequency/
def SortNumbersByFrequenceyV1(nums: List[int]) -> List[int]:
    n = len(nums)
    counts = Counter(nums)
    buckets = [[] for _ in range(n+1)]
    for char, count in counts.items():
        buckets[count].append(char)
    result = []
    for i in range(1, n+1):
        if buckets[i]:
            buckets[i].sort(reverse=True)
            for num in buckets[i]:
                result.extend([num] * i) 
    return result

def SortNumbersByFrequenceyV2(nums: List[int]) -> List[int]:
    r = Counter(nums)
    return sorted(nums, key=lambda x: (r[x], -x))


##############################################################################################
###   [CONCURRENCY]
##############################################################################################


# https://leetcode.com/problems/print-in-order/
class Foo:
    def __init__(self):
        self.sleeptime = 0.001
        self.semaphore = 0

    def first(self, printFirst: 'Callable[[], None]') -> None:
        printFirst()
        self.semaphore += 1

    def second(self, printSecond: 'Callable[[], None]') -> None:
        while self.semaphore != 1:
            time.sleep(self.sleeptime)
        printSecond()
        self.semaphore += 1

    def third(self, printThird: 'Callable[[], None]') -> None:
        while self.semaphore != 2:
            time.sleep(self.sleeptime)
        printThird()


# https://leetcode.com/problems/print-foobar-alternately/
class FooBar:
    def __init__(self, n):
        self.n = n
        self.semaphore = 0

    def foo(self, printFoo: 'Callable[[], None]') -> None:
        for _ in range(self.n):
            while self.semaphore == 1:
                time.sleep(0.001)
            printFoo()
            self.semaphore = 1

    def bar(self, printBar: 'Callable[[], None]') -> None:
        for _ in range(self.n):
            while self.semaphore == 0:
                time.sleep(0.001)
            printBar()
            self.semaphore = 0