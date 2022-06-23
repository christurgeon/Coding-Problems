#   DS
# https://igotanoffer.com/blogs/tech/data-structure-interview-questions
#   ALGO
# https://igotanoffer.com/blogs/tech/algorithms-interview-questions

"""
    For a given problem that lends itself to be solved naturally recursively (i.e. N-Queens)
        if you want optimal solution dynamic programming 
        if you want all solutions, backtracking is used
"""

from asyncio import QueueEmpty
from email.base64mime import header_length
import heapq
import collections
from typing import *
from functools import lru_cache

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
###   ARRAYS
##############################################################################################


def ConcatenateTwoLists(nums: List[int]) -> List[int]:
    return nums + nums


# https://leetcode.com/problems/contains-duplicate/
def ContainsDuplicate(nums: List[int]) -> bool:
    n = len(nums)
    return n != len(set(nums))


# https://leetcode.com/problems/arranging-coins/
def ArrangeCoins(n):
    stairs = 1
    while n - stairs >= 0:
        n -= stairs
        stairs += 1
    return stairs - 1      


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


def LongestConsecutiveInAnArray(nums: List[int]) -> int:
    if not nums:
        return 0
    
    s = set()
    for num in nums:
        s.add(num)
    
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


def BestTimeToBuyAndSellStock(prices: List[int]) -> int:
    profit = 0
    running_min = prices[0]
    for price in prices:
        profit = max(profit, price - running_min)
        running_min = min(running_min, price)
    return profit


def MergeTwoSortedLists(A, B):
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


def CountTheFrequencyOfAnElementInAnArray(A):
    d = dict()
    for element in A:
        if element not in d:
            d[element] = 1
        else:
            d[element] += 1
    return d


def MoveAllZerosToBeginningOfArray(A):
    write_index = 0
    for i in range(1, len(A)):
        if A[i] == 0:
            A[write_index], A[i] = A[i], A[write_index]
            write_index += 1
    return A


def MoveAllZerosToEndOfArray(A):
    write_index = len(A) - 1
    for i in reversed(range(len(A))):
        if A[i] == 0:
            A[write_index], A[i] = A[i], A[write_index]
            write_index -= 1
    return A


def BinarySearchAnArray(A, x):
    low, high = 0, len(A) - 1

    while low <= high:
        mid = (low + high) // 2 # average of low and high
        
        if A[mid] > x:
            high = mid - 1
        elif A[mid] < x:
            low = mid + 1
        else:
            return True

    return False


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


def CountDecreasingSubArrays(A):
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
    """
    Do not return anything, modify matrix in-place instead.
    """
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
        ROWS, COLS = len(matrix), len(matrix[0])
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
def topKFrequent(nums: List[int], k: int) -> List[int]:
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


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
def MaxProfit(prices: List[int]) -> int:
    result = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            result += prices[i] - prices[i-1]
    return result


##############################################################################################
###   STRING
##############################################################################################


def LengthOfLastWord(s: str) -> int:
    size = 0
    for char in reversed(s):
        if char == " ":
            if size > 0:
                return size 
        else:
            size += 1
    return size


"""
Given a string consisting of only a's and b's what is the minimum 
amount of palindromes you can remove to make it empty
    solution: since its two chars if s is not palindrome, 
              then it takes 2 removals only
"""
def RemovePalindromicString(s: str) -> int:
    if s == s[::-1]:
        return 1
    return 2


def RemoveVowelsFromAString(S):
    vowels = dict.fromkeys(["a", "e", "i", "o", "u"])
    new_string = []
    for c in S:
        if c not in vowels:
            new_string.append(c)
    return ''.join(new_string)


def DefangingAnIPAddress(IP):
    split_IP = IP.split('.')
    return "[.]".join(split_IP)


# find how many of the stones you have are also jewels
def JewelsAndStones(jewels, stones):
    d = dict.fromkeys(list(jewels))
    cnt = 0
    for stone in stones:
        if stone in d:
            cnt += 1
    return cnt


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


def FirstUniqueChar(s: str) -> int:
    d = {}
    for char in s:
        d[char] = 1 if char not in d else d[char] + 1

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


def MinimumWindowSubstring(s, target):
    counts = {}
    for t in target:
        counts[t] = 1 if t not in counts else counts[t] + 1
        
    start, found = 0, 0
    minL, minR = float('-inf'), float('inf')
    min_window = ""
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


# https://leetcode.com/problems/increasing-decreasing-string/submissions/
def sortString(s: str) -> str:
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


##############################################################################################
###   LISTS
##############################################################################################


def DeleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    current = head
    while current:
        while current.next and current.val == current.next.val:
            current.next = current.next.next
        current = current.next
    return head


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
        

# https://leetcode.com/problems/reverse-linked-list/
def ReverseLinkedList(head: Optional[ListNode]) -> Optional[ListNode]:
    previous = None
    current = head
    while current:
        temp = current.next
        current.next = previous
        previous, current = current, temp
    return previous

def ReverseLinkedListRecursive(head: Optional[ListNode]) -> Optional[ListNode]:
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
###   STACKS AND QUEUES
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


# LeetCode Version
def evalRPN(tokens: List[str]) -> int:
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
        

def ValidParenthesis(S):
    lookup = { '{': '}', '[': ']', '(': ')' }
    stack = []
    for c in S:
        if c in lookup:
            stack.append(lookup[c])
        elif len(stack) == 0 or c != stack.pop():
            return False 
    return len(stack) == 0


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


# use two queues to implement a stack
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


def BasicCalculator():
    pass
def LongestValidParentheses():
    pass 
def TrappingRainWater():
    pass 
def LargestRectangleInHistogram():
    pass


##############################################################################################
###   TREES
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

def BinaryTreePrerderTraversal(root):
    if root:
        return  [root.val] + BinaryTreePrerderTraversal(root.left) + BinaryTreePrerderTraversal(root.right)
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


"""
    Search for BST violations in a BFS manner, store upper and lower bound
    on the keys stored at the subtree rooted at that node.
"""
def IsBinaryTreeABinarySearchTree(tree):
    queue = deque((tree, float('-inf'), float('inf')))
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


"""
    Traverse down the right subtree and whenever end is reached, we know that is the greatest value
"""
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


def SymmetricTree(root):
    def isSym(left, right):
        if not left and not right:
            return True
        elif left and right:
            return left.val == right.val and isSym(left.left, right.right) and isSym(left.right, right.left)
        else:
            return False
    return not root or str.isdigit(root.left, root.right) 
    

def MaximumDepthOfBinaryTree(root):
    def traverse(node, depth):
        if node:
            return max(traverse(node.left, depth + 1), traverse(node.right, depth + 1))
        return depth
    return traverse(root, 0)


# iterative example from leetcode discussion board, for reference 
def maxDepth(root):
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


def ValidateBinarySearchTree(root):
    if not root:
        return True 
    if root.left and root.left.val >= root.val:
        return False
    if root.right and root.right.val <= root.val:
        return False
    return ValidateBinarySearchTree(root.left) and ValidateBinarySearchTree(root.right)


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


class Solution:
    def __init__(self):
        self.max_path_sum = float('-inf')

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # i think we have to traverse all O(n) time, let's try recursively
        
        # do bottom up search and trim paths
        def search(node):
            if not node:
                return 0
            l = search(node.left)
            r = search(node.right)
            
            # update max
            self.max_path_sum = max(self.max_path_sum, l + node.val + r)
            
            # greatest path from bottom up to current node
            bottom_up_max_sum = node.val + max(l, r)
            return max(bottom_up_max_sum, 0)
        
        search(root)
        return self.max_path_sum    


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

def maxAncestorDiff(root: Optional[TreeNode]) -> int:
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
@lru_cache(None)
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
        

# EPI
"""
def RebuildBSTFromPreorder(preorder_sequence: List[int]):
    def helper(lower, upper):
        pass 
    pass
def FindClosestElementsInSortedArrays(sorted_arrays: List[List[int]]):
    pass 
def BuildMinHeightBSTFromSortedArray(array: List[int]):
    pass 
def PairIncludesAncestorAndDescendantOf_M(possible_anc_or_desc_0, possible_anc_or_desc_0, middle):
    pass
def RangeLookupInBST(tree, interval):
    pass
def SerializeAndDeserializeBinaryTree():
    pass 
def BinaryTreeCameras():
    pass 
"""


##############################################################################################
###   GRAPHS
##############################################################################################


def PathExists(edges: List[List[int]], source: int, destination: int) -> bool:
    if source == destination:
        return True
    
    graph = {}
    for edge in edges:
        if edge[0] not in graph:
            graph[edge[0]] = []
        if edge[1] not in graph:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
        
    visited, stack = {}, [source]
    while stack:
        current = stack.pop()
        if current in graph:
            for neighbor in graph[current]:
                if neighbor == destination:
                    return True
                if neighbor not in visited:
                    visited[neighbor] = None 
                    stack.append(neighbor)
                
    return False


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
    # create adjacency list and sort the destinations lexigraphically 
    graph = {}
    for ticket in tickets:
        departure, destination = ticket[0], ticket[1]
        if departure not in graph:
            graph[departure] = []
        if destination not in graph:
            graph[destination] = []
        graph[departure].append(destination)
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
        print(stack)
            
    return result[::-1]


# https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/submissions/
"""Could also do a topological sort, need to review that"""
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


# https://leetcode.com/problems/all-paths-from-source-to-target/submissions/
def AllPathsFromFirstNodeToLast(graph: List[List[int]]) -> List[List[int]]:
    target = len(graph) - 1
    paths = []
    queue = deque([ [0, [0]] ])
    while queue:
        current, path_so_far = queue.popleft()
        if current == target:
            paths.append(path_so_far)
        else:
            for neighbor in graph[current]:
                queue.append([neighbor, path_so_far + [neighbor]])
    return paths


##############################################################################################
###   MAPS
##############################################################################################


# for distinct nums 
def TwoSum(nums, target):
    d = {value: idx for idx, value in enumerate(nums)}
    for value, idx in d.items():
        other = target - value
        if other in d:
            return [idx, d[target-value]]
    return []


# not distinct
def TwoSumsNotDistinct(nums, target):
    d = dict()
    for idx, value in enumerate(nums):
        if value in d:
            return [idx, d[value]]
        diff = target - value 
        d[diff] = idx
    return []


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
def MajorityElement1(self, nums: List[int]) -> int:
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


def GroupAnagrams(strs):
    result = {}
    for s in strs:
        transformed_s = ''.join(sorted(s))
        if transformed_s in result:
            result[transformed_s] += [s]
        else:
            result[transformed_s] = [s]
    return result.values()


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


##############################################################################################
###   HEAPS
##############################################################################################

# heapify operation is actually O(n)

# remember that heapq only applies minheap functionality
# negate all values to get maxheap functionality

def KthLargestElementInAStream(self, k, nums):
    self.k = k
    self.running = nums[:k]
    heapq.heapify(self.running)
    for i in range(k, len(nums)):
        heapq.heappushpop(self.running, nums[i])

def add(self, val):
    heapq.heappush(self.running, val)
    if len(self.running) > self.k:
        heapq.heappop(self.running)
    return self.running[0]


def FindKthLargest(nums: List[int], k: int) -> int:
    running = nums[:k]
    heapq.heapify(running)
    for i in range(k, len(nums)):
        heapq.heappushpop(running, nums[i])
    return heapq.heappop(running)


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


def ReorganizeString(self, s: str) -> str:
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


##############################################################################################
###   DEPTH-FIRST SEARCH
##############################################################################################


def SameTree(lhead: TreeNode, rhead: TreeNode):
    if not lhead and not rhead:
        return True
    if (not lhead) ^ (not rhead):
        return False 
    if lhead.val != rhead.val:
        return False
    return SameTree(lhead.left, rhead.left) and SameTree(lhead.right, rhead.right) 


"""
Surrounded regions should not be on the border, which means that any 'O' on the border of the 
board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to 
an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent 
cells connected horizontally or vertically.

key is that border by default is not surrounded and any they touch will be not surrounded
    - set these to T then keep them keep them as O at the end
    - all other O's which were not reached can assume to be converted to X
"""
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

    
def NumberOfIslands(self, grid: List[List[str]]) -> int:
        
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


# https://leetcode.com/problems/course-schedule/
def CourseSchedule(intervals: List[List[int]], newInterval: List[int]):
    i, result = 0, []
    while (i < len(intervals)) and (newInterval[0] > intervals[i][1]):
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
    Could also just traverse and compare parent to child and if there is a difference than return false
    might be a bit quicker XD
"""
def IsSingleValTree(root: Optional[TreeNode]) -> bool:
    def traverse(node, value):
        if node:
            if node.val != value:
                return False
            return traverse(node.left, value) and traverse(node.right, value)
        return True
    return traverse(root, root.val)


# N-QUEENS
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
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
        
        def validPlacement(r, c):
            # we know it is not within same row, so just check column and diagonals       
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
    def solveNQueens(self, n: int) -> List[List[str]]:
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


##############################################################################################
###   BREADTH-FIRST SEARCH
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
        

# https://leetcode.com/problems/max-area-of-island/submissions/
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


def HIndex(citations: List[int]) -> int:
    n = len(citations)
    citations.sort()
    for i, citation in enumerate(citations):
        if citation >= n - i:
            return n - i
    return 0


def AddDigitsMathematical(num):
    return 0 if num == 0 else (num - 1) % 9 + 1


def AddDigitsLoop(num):
    while num > 9:
        num = sum(int(d) for d in str(num))
    return num


##############################################################################################
###   DYNAMIC PROGRAMMING
##############################################################################################


""" NOTES
typically want to use @lru_cache(None) to cache recursive results when they are likely to come 
up again with same arguements

useful for fibonacci can set lru_cache(2) so it will always cache past two results
"""
def MaximumSubarray(nums):
    dp = [0] * len(nums)
    dp[0] = nums[0]
    running_max = dp[0]
    for i in range(1, len(nums)):
        dp[i] = max(nums[i], dp[i-1] + nums[i])
        running_max = max(running_max, dp[i])
    return running_max

def MaximumSubarrayV2(nums):
    max_seen = max_end = 0
    for num in nums:
        max_end = max(num, num + max_end)
        max_seen = max(max_seen, max_end)
    return max_seen


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


def CanInterleaveString(self, s1: str, s2: str, s3: str) -> bool:
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
def FindTargetSumWays(self, nums: List[int], target: int) -> int:
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


##############################################################################################
###   MISC
##############################################################################################


def HammingWeight(self, n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


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


##############################################################################################
###   GREEDY & INVARIANTS
##############################################################################################

""" NOTES 
1. A greedy algo is often right choice for optimization problems where there's a natural set of choices to choose from
2. Often easier to conceptualize recursively, then implement iteratively for better performance
3. Even if greedy does not give optimal results, can give insights into optimum or serve as heuristic
"""

# https://leetcode.com/problems/assign-cookies/submissions/
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


# https://leetcode.com/problems/can-place-flowers/submissions/
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


# https://leetcode.com/problems/lemonade-change/submissions/
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


# each worker assigned exactly 2 tasks, each one takes fixed amount of time, and tasks are ind.
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


def MajorityElementSearch(stream: Iterator[int]) -> str:
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


##############################################################################################
###   SORTING
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